"""Compare traditional and Balanced KMeans IVF-Flat indexes on GloVe/GIST.

The existing ``TenVS_IVFFlat_nlist1024.bin`` files are treated as the
traditional-Lloyd-KMeans baseline.  This script builds only the additional
Balanced KMeans index, evaluates both indexes, writes machine-readable
results, and generates one two-panel figure per dataset.

Results are written to ``bench/results`` and the figures are written as PDF
files to ``plot/Figures`` by default.

Example:

    python plot/benchmark_kmeans_ivfflat.py

To use the nprobe values from the plot specification literally, run:

    python plot/benchmark_kmeans_ivfflat.py --nprobe 50 100 150
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

N_LIST = 1024
N_ITER = 25
SEED = 1234
N_INIT = 1
TOPK = 10
DEFAULT_NPROBES = (50, 100, 150)
DEFAULT_QUERY_BATCH = 64
DEFAULT_DEVICE = "cuda:0"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "bench" / "results"
DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "Figures"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    base_candidates: tuple[Path, ...]
    query_candidates: tuple[Path, ...]
    groundtruth_candidates: tuple[Path, ...]

    @property
    def traditional_index(self) -> Path:
        return self.root / "GPUIndex" / f"TenVS_IVFFlat_nlist{N_LIST}.bin"

    @property
    def balanced_index(self) -> Path:
        return self.root / "GPUIndex" / f"TenVS_IVFFlat_BalancedKMeans_nlist{N_LIST}.bin"


DATASETS = (
    DatasetConfig(
        name="glove1.2m",
        root=Path("/storage/vector_data/glove1.2m"),
        base_candidates=(Path("glove1.2m_base.fvecs"),),
        query_candidates=(Path("glove1.2m_query.fvecs"),),
        groundtruth_candidates=(Path("glove1.2m_groundtruth.ivecs"),),
    ),
    DatasetConfig(
        name="gist",
        root=Path("/storage/vector_data/gist"),
        # Both layouts have existed on the shared storage over time.
        base_candidates=(Path("gist_base.fvecs"), Path("gist/gist_base.fvecs")),
        query_candidates=(Path("gist_query.fvecs"), Path("gist/gist_query.fvecs")),
        groundtruth_candidates=(
            Path("gist_groundtruth.ivecs"),
            Path("gist/gist_groundtruth.ivecs"),
        ),
    ),
)


def first_existing(root: Path, candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    attempted = ", ".join(str(root / candidate) for candidate in candidates)
    raise FileNotFoundError(f"None of the candidate files exists: {attempted}")


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "IVF-Flat search in this repository requires CUDA/Triton; "
            f"got device={device}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices exist"
        )


def build_balanced_index(
    config: DatasetConfig,
    *,
    device: torch.device,
    force: bool,
    max_points_per_centroid: int | None,
) -> Path:
    """Build the Balanced KMeans IVF-Flat index if it is not already present."""
    from flash_kmeans import fit_best_kmeans
    from tensorvs import IVFFlatBuildResult, load_dataset, save_ivfflat_index

    output_path = config.balanced_index
    if output_path.exists() and not force:
        print(f"[{config.name}] Balanced index exists: {output_path}")
        return output_path

    base_path = first_existing(config.root, config.base_candidates)
    print(f"[{config.name}] loading base vectors from {base_path}")
    vectors = load_dataset(base_path)
    print(
        f"[{config.name}] building Balanced KMeans index: "
        f"vectors={tuple(vectors.shape)}, nlist={N_LIST}, device={device}"
    )

    started = time.perf_counter()
    with torch.inference_mode():
        trained = fit_best_kmeans(
            vectors,
            d=int(vectors.shape[1]),
            k=N_LIST,
            niter=N_ITER,
            use_triton=True,
            device=device,
            seed=SEED,
            n_init=N_INIT,
            stage=0,
            max_points_per_centroid=max_points_per_centroid,
            # This is the cuVS-style hierarchical balanced builder used for
            # coarse IVF quantizers in this repository.
            algorithm="balanced",
            balanced_hierarchical=True,
        )

    build_result = IVFFlatBuildResult(
        vectors=vectors,
        labels=trained.labels,
        centroids=trained.centroids,
        device=device,
        seed=SEED,
        n_init=N_INIT,
        coarse_inertia=trained.inertia,
        coarse_seed=trained.seed,
    )
    elapsed = time.perf_counter() - started
    print(f"[{config.name}] Balanced KMeans training finished in {elapsed:.2f}s")
    print(f"[{config.name}] saving Balanced index to {output_path}")
    save_ivfflat_index(
        build_result,
        output_path,
        dataset_path=base_path,
        build_time_seconds=elapsed,
    )

    # Do not keep a several-GB base matrix alive while the saved archive is
    # loaded for the search benchmark.
    del build_result, trained, vectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def list_statistics(list_count: torch.Tensor) -> dict[str, Any]:
    counts = list_count.detach().cpu().to(torch.int64)
    if counts.ndim != 1 or counts.numel() != N_LIST:
        raise ValueError(
            f"Expected {N_LIST} one-dimensional list counts, got {tuple(counts.shape)}"
        )
    values = counts.numpy().astype(np.int64, copy=False)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        # Keep the complete distribution so plot-only runs do not need to
        # deserialize a multi-GB IVF-Flat archive again.
        "list_lengths": values.tolist(),
    }


def search_recall(
    index: Any,
    queries: torch.Tensor,
    groundtruth: torch.Tensor,
    *,
    nprobes: tuple[int, ...],
    query_batch: int,
    device: torch.device,
    method: str,
) -> dict[str, float]:
    """Evaluate Recall@TOPK for all requested nprobe values."""
    from tensorvs import compute_recall_at_k, search_ivfflat_index

    if queries.shape[0] != groundtruth.shape[0]:
        raise ValueError(
            f"Query/groundtruth row mismatch: {queries.shape[0]} vs {groundtruth.shape[0]}"
        )

    index.reside_on_device(device)
    recalls: dict[str, float] = {}
    q_batch = max(1, int(query_batch))
    for nprobe in nprobes:
        retrieved_ids: list[torch.Tensor] = []
        started = time.perf_counter()
        for start in range(0, queries.shape[0], q_batch):
            end = min(start + q_batch, queries.shape[0])
            batch = queries[start:end]
            actual_batch_size = int(batch.shape[0])
            if actual_batch_size < q_batch:
                batch = torch.cat((batch, batch[-1:].expand(q_batch - actual_batch_size, -1)))
            with torch.inference_mode():
                _, batch_ids = search_ivfflat_index(
                    index,
                    batch,
                    topk=TOPK,
                    nprobe=int(nprobe),
                    device=device,
                    selection_mode="triton_topk",
                )
            retrieved_ids.append(batch_ids[:actual_batch_size].cpu())

        all_ids = torch.cat(retrieved_ids, dim=0)
        recall = compute_recall_at_k(all_ids, groundtruth, TOPK)
        recalls[str(nprobe)] = float(recall)
        print(
            f"  {method} "
            f"nprobe={nprobe}: Recall@{TOPK}={recall:.6f} "
            f"({time.perf_counter() - started:.2f}s)"
        )
    return recalls


def evaluate_index(
    index_path: Path,
    *,
    queries: torch.Tensor,
    groundtruth: torch.Tensor,
    nprobes: tuple[int, ...],
    query_batch: int,
    device: torch.device,
    method: str,
) -> dict[str, Any]:
    from tensorvs import load_ivfflat_index

    print(f"Loading {method} index: {index_path}")
    index = load_ivfflat_index(index_path, load_lists=True)
    if int(index.centroids.shape[0]) != N_LIST:
        raise ValueError(f"{index_path} has nlist={index.centroids.shape[0]}, expected {N_LIST}")
    stats = list_statistics(index.list_count)
    print(
        f"  list length min={stats['min']}, mean={stats['mean']:.3f}, "
        f"max={stats['max']}"
    )
    # The old archives do not contain a KMeans-method metadata field, so the
    # explicit method argument is authoritative for the benchmark output.
    recalls = search_recall(
        index,
        queries,
        groundtruth,
        nprobes=nprobes,
        query_batch=query_batch,
        device=device,
        method=method,
    )
    del index
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "method": method,
        "index_path": str(index_path),
        "list_stats": stats,
        "recall_at_10": recalls,
    }


def plot_dataset_result(result: dict[str, Any], output_path: Path, nprobes: tuple[int, ...]) -> None:
    """Create panel A (list-size boxplots) and panel B (Recall bars)."""
    methods = ("traditional_kmeans", "balanced_kmeans")
    display_names = {"traditional_kmeans": "Traditional KMeans", "balanced_kmeans": "Balanced KMeans"}
    colors = {"traditional_kmeans": "#4C78A8", "balanced_kmeans": "#F58518"}

    # A wide, shallow layout is easier to place in a paper and leaves room
    # below each panel for the requested subfigure captions.
    fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(14, 3.8))
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.27,
        top=0.84,
        wspace=0.18,
    )

    box_data = [result["indexes"][method]["list_stats"]["list_lengths"] for method in methods]
    box = ax_box.boxplot(
        box_data,
        tick_labels=[display_names[method] for method in methods],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(1.0)
    for x, method in enumerate(methods, start=1):
        stats = result["indexes"][method]["list_stats"]
        ax_box.scatter(
            x,
            stats["min"],
            marker="v",
            color="#E41A1C",
            edgecolors="#111111",
            linewidths=0.8,
            s=100,
            zorder=3,
            label="min" if x == 1 else None,
        )
        ax_box.scatter(
            x,
            stats["max"],
            marker="^",
            color="#E41A1C",
            edgecolors="#111111",
            linewidths=0.8,
            s=100,
            zorder=3,
            label="max" if x == 1 else None,
        )
        ax_box.annotate(
            f"mean={stats['mean']:.1f}",
            (x, stats["mean"]),
            xytext=(7, 0),
            textcoords="offset points",
            fontsize=10,
            va="center",
        )
    ax_box.set_ylabel("List length")
    ax_box.grid(axis="y", alpha=0.25)
    ax_box.legend(loc="upper right", fontsize=8)
    ax_box.text(
        0.5,
        -0.24,
        "(a) Distribution of IVF List Lengths",
        transform=ax_box.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    x = np.arange(len(nprobes))
    width = 0.36
    for offset, method in ((-width / 2, "traditional_kmeans"), (width / 2, "balanced_kmeans")):
        values = [result["indexes"][method]["recall_at_10"][str(nprobe)] for nprobe in nprobes]
        bars = ax_bar.bar(
            x + offset,
            values,
            width,
            label=display_names[method],
            color=colors[method],
        )
        for bar, value in zip(bars, values):
            # Keep labels for near-perfect Recall inside the axes and inside
            # the bar, so they do not cross the y=1.0 upper border.
            if value >= 0.985:
                # There is not enough room above values this close to one
                # when the y-axis maximum is fixed at 1.0.
                label_y = value - 0.015
                vertical_alignment = "top"
            else:
                label_y = value + 0.005
                vertical_alignment = "bottom"
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:.4f}",
                ha="center",
                va=vertical_alignment,
                fontsize=7,
            )
    ax_bar.set_xlabel("nprobe")
    ax_bar.set_ylabel(f"Recall@{TOPK}")
    ax_bar.set_xticks(x, [str(value) for value in nprobes])
    # Recall is bounded by one. Starting at 0.4 makes the differences between
    # the methods easier to read without changing the actual values.
    ax_bar.set_ylim(0.6, 1.0)
    ax_bar.set_yticks((0.6, 0.8, 1.0))
    ax_bar.grid(axis="y", alpha=0.25)
    # Keep the method legend inside panel B.
    ax_bar.legend(fontsize=8, loc="upper left", ncol=1)
    ax_bar.text(
        0.5,
        -0.24,
        "(b) Comparison of Achievable Recalls",
        transform=ax_bar.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_results(results: list[dict[str, Any]], output_dir: Path, nprobes: tuple[int, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "nlist": N_LIST,
        "topk": TOPK,
        "nprobes": list(nprobes),
        "results": results,
    }
    json_path = output_dir / "kmeans_ivfflat_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_lines = ["dataset,method,min_list_length,mean_list_length,max_list_length," + ",".join(f"recall_nprobe_{n}" for n in nprobes)]
    for result in results:
        for method in ("traditional_kmeans", "balanced_kmeans"):
            item = result["indexes"][method]
            stats = item["list_stats"]
            recalls = item["recall_at_10"]
            row = [
                result["dataset"],
                method,
                str(stats["min"]),
                f"{stats['mean']:.8f}",
                str(stats["max"]),
                *(f"{recalls[str(n)]:.8f}" for n in nprobes),
            ]
            csv_lines.append(",".join(row))
    (output_dir / "kmeans_ivfflat_results.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Results written to {json_path} and {output_dir / 'kmeans_ivfflat_results.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[config.name for config in DATASETS] + ["all"],
        default="glove1.2m",
        help="Dataset to build/evaluate/plot (default: glove1.2m)",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="CUDA device, e.g. cuda:0 or cuda:6")
    parser.add_argument("--query-batch", type=int, default=DEFAULT_QUERY_BATCH)
    parser.add_argument("--nprobe", type=int, nargs="+", default=list(DEFAULT_NPROBES))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--skip-build", action="store_true", help="Do not build the Balanced index")
    parser.add_argument("--force-build", action="store_true", help="Overwrite an existing Balanced index")
    parser.add_argument(
        "--max-points-per-centroid",
        type=int,
        default=None,
        help="KMeans training sampling cap; default None matches the old full-data baseline",
    )
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from the saved JSON results")
    return parser.parse_args()


def plot_only(results_dir: Path, figures_dir: Path, dataset_name: str) -> None:
    payload = json.loads((results_dir / "kmeans_ivfflat_results.json").read_text(encoding="utf-8"))
    nprobes = tuple(int(value) for value in payload["nprobes"])
    for result in payload["results"]:
        if dataset_name != "all" and result["dataset"] != dataset_name:
            continue
        path = figures_dir / f"{result['dataset']}_kmeans_ivfflat.pdf"
        plot_dataset_result(result, path, nprobes)
        print(f"Figure written to {path}")


def main() -> None:
    args = parse_args()
    if args.plot_only:
        plot_only(args.results_dir, args.figures_dir, args.dataset)
        return

    if any(int(nprobe) <= 0 for nprobe in args.nprobe):
        raise ValueError("All nprobe values must be positive")
    if args.query_batch <= 0:
        raise ValueError("--query-batch must be positive")

    device = torch.device(args.device)
    validate_device(device)
    selected = DATASETS if args.dataset == "all" else tuple(config for config in DATASETS if config.name == args.dataset)
    nprobes = tuple(int(value) for value in args.nprobe)
    results: list[dict[str, Any]] = []

    for config in selected:
        if not args.skip_build:
            build_balanced_index(
                config,
                device=device,
                force=args.force_build,
                max_points_per_centroid=args.max_points_per_centroid,
            )

        from tensorvs import load_dataset, load_ivecs

        base_path = first_existing(config.root, config.base_candidates)
        query_path = first_existing(config.root, config.query_candidates)
        groundtruth_path = first_existing(config.root, config.groundtruth_candidates)
        if not config.traditional_index.exists():
            raise FileNotFoundError(f"Traditional index not found: {config.traditional_index}")
        if not config.balanced_index.exists():
            raise FileNotFoundError(f"Balanced index not found: {config.balanced_index}")

        print(f"[{config.name}] loading queries and groundtruth")
        queries = load_dataset(query_path)
        groundtruth = load_ivecs(groundtruth_path)
        print(
            f"[{config.name}] queries={tuple(queries.shape)}, "
            f"groundtruth={tuple(groundtruth.shape)}"
        )

        dataset_result = {
            "dataset": config.name,
            "base_path": str(base_path),
            "query_path": str(query_path),
            "groundtruth_path": str(groundtruth_path),
            "indexes": {},
        }
        dataset_result["indexes"]["traditional_kmeans"] = evaluate_index(
            config.traditional_index,
            queries=queries,
            groundtruth=groundtruth,
            nprobes=nprobes,
            query_batch=args.query_batch,
            device=device,
            method="traditional KMeans",
        )
        dataset_result["indexes"]["balanced_kmeans"] = evaluate_index(
            config.balanced_index,
            queries=queries,
            groundtruth=groundtruth,
            nprobes=nprobes,
            query_batch=args.query_batch,
            device=device,
            method="Balanced KMeans",
        )
        results.append(dataset_result)

        figure_path = args.figures_dir / f"{config.name}_kmeans_ivfflat.pdf"
        plot_dataset_result(dataset_result, figure_path, nprobes)
        print(f"[{config.name}] figure written to {figure_path}")

        del queries, groundtruth
        gc.collect()
        torch.cuda.empty_cache()

    write_results(results, args.results_dir, nprobes)


if __name__ == "__main__":
    main()
