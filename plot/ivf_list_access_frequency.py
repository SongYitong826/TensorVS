"""Measure and plot how often IVF lists are selected by real queries.

The experiment counts one access whenever an IVF list appears in a query's
coarse ``nprobe`` selection.  It uses the centroids from the saved TensorVS
IVF-Flat indexes, so the measurement matches the lists used by search.

By default the script evaluates all 1,000 official queries from Hunyuan and
OpenAI-1536 with ``nprobe=64`` and ``nlist=1024``::

    conda run -n tensorvs python plot/ivf_list_access_frequency.py

The figure, per-list CSV files, and a JSON summary are written to
``plot/Figures`` and ``plot/results``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MaxNLocator
import numpy as np
import torch

from tensorvs import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/storage/vector_data")
DEFAULT_FIGURE_DIR = SCRIPT_DIR / "Figures"
DEFAULT_RESULT_DIR = SCRIPT_DIR / "results"
DEFAULT_NPROBE = 64
DEFAULT_QUERY_BATCH_SIZE = 256


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    query_filename: str
    index_filename: str = "TenVS_IVFFlat_nlist1024.bin"


@dataclass(frozen=True)
class AccessSummary:
    dataset: str
    label: str
    index_path: str
    query_path: str
    num_database_vectors: int | None
    dimension: int
    nlist: int
    nprobe: int
    num_queries: int
    total_accesses: int
    nonzero_lists: int
    min_access_count: int
    median_access_count: float
    mean_access_count: float
    max_access_count: int
    max_over_mean: float
    coefficient_of_variation: float
    gini_coefficient: float
    top_1_percent_access_share: float
    top_10_percent_access_share: float


DATASETS = (
    DatasetConfig("hunyuan", "Hunyuan", "hunyuan_query.fvecs"),
    DatasetConfig("openai1536", "OpenAI", "openai1536_query.fvecs"),
)


def _load_index_centroids(index_path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load only the small coarse-quantizer state needed by this experiment.

    TensorVS single-file indexes contain the full vector layout.  ``mmap=True``
    keeps those large, unused storages out of resident memory while the
    centroids are copied into a small standalone tensor.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"IVF index not found: {index_path}")

    if index_path.is_file():
        payload = torch.load(
            index_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if not isinstance(payload, dict) or "centroids" not in payload:
            raise ValueError(f"Invalid TensorVS IVF index: {index_path}")
        centroids = payload["centroids"].to(torch.float32).clone().contiguous()
        metadata = dict(payload.get("metadata", {}))
        del payload
        return centroids, metadata

    centroids_path = index_path / "centroids.pt"
    metadata_path = index_path / "metadata.json"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Index centroids not found: {centroids_path}")
    centroids = (
        torch.load(centroids_path, map_location="cpu", weights_only=True)
        .to(torch.float32)
        .contiguous()
    )
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return centroids, metadata


def count_list_accesses(
    queries: torch.Tensor,
    centroids: torch.Tensor,
    *,
    nprobe: int,
    query_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return one coarse-selection access count for every IVF list."""
    if queries.ndim != 2 or centroids.ndim != 2:
        raise ValueError("queries and centroids must both be two-dimensional")
    if queries.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"Query dimension {queries.shape[1]} does not match centroid "
            f"dimension {centroids.shape[1]}"
        )
    if not 1 <= nprobe <= centroids.shape[0]:
        raise ValueError(f"nprobe must be in [1, {centroids.shape[0]}], got {nprobe}")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")

    centroids_device = centroids.to(device=device, dtype=torch.float32)
    centroid_norms = centroids_device.square().sum(dim=1).unsqueeze(0)
    counts = torch.zeros(centroids.shape[0], dtype=torch.int64, device=device)

    # Match TensorVS coarse selection: use IEEE FP32 rather than TF32 on CUDA.
    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            for query_batch in queries.split(query_batch_size):
                query_batch = query_batch.to(device=device, dtype=torch.float32)
                # The omitted ||q||^2 is constant across all centroids and
                # cannot change which lists are selected.
                coarse_scores = torch.addmm(
                    centroid_norms,
                    query_batch,
                    centroids_device.transpose(0, 1),
                    beta=1.0,
                    alpha=-2.0,
                )
                probes = coarse_scores.topk(
                    nprobe, dim=1, largest=False, sorted=False
                ).indices
                counts.add_(
                    torch.bincount(probes.flatten(), minlength=centroids.shape[0])
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32

    return counts.cpu()


def _gini(values: np.ndarray) -> float:
    values = np.sort(values.astype(np.float64, copy=False))
    total = float(values.sum())
    if total == 0.0:
        return 0.0
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2.0 * np.dot(ranks, values) / total - values.size - 1) / values.size)


def summarize_accesses(
    config: DatasetConfig,
    counts: torch.Tensor,
    *,
    index_path: Path,
    query_path: Path,
    metadata: dict[str, Any],
    dimension: int,
    nprobe: int,
    num_queries: int,
) -> AccessSummary:
    values = counts.numpy().astype(np.float64, copy=False)
    descending = np.sort(values)[::-1]
    mean = float(values.mean())

    def top_share(fraction: float) -> float:
        number = max(1, math.ceil(values.size * fraction))
        return float(descending[:number].sum() / descending.sum())

    return AccessSummary(
        dataset=config.key,
        label=config.label,
        index_path=str(index_path),
        query_path=str(query_path),
        num_database_vectors=(
            int(metadata["num_vectors"]) if metadata.get("num_vectors") is not None else None
        ),
        dimension=dimension,
        nlist=int(values.size),
        nprobe=nprobe,
        num_queries=num_queries,
        total_accesses=int(values.sum()),
        nonzero_lists=int(np.count_nonzero(values)),
        min_access_count=int(values.min()),
        median_access_count=float(np.median(values)),
        mean_access_count=mean,
        max_access_count=int(values.max()),
        max_over_mean=float(values.max() / mean),
        coefficient_of_variation=float(values.std() / mean),
        gini_coefficient=_gini(values),
        top_1_percent_access_share=top_share(0.01),
        top_10_percent_access_share=top_share(0.10),
    )


def write_access_csv(
    output_path: Path,
    config: DatasetConfig,
    counts: torch.Tensor,
    num_queries: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_list_ids = torch.argsort(counts, descending=True, stable=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "dataset",
                "rank",
                "list_id",
                "access_count",
                "query_access_rate",
            ),
        )
        writer.writeheader()
        for rank, list_id in enumerate(ranked_list_ids.tolist(), start=1):
            access_count = int(counts[list_id])
            writer.writerow(
                {
                    "dataset": config.key,
                    "rank": rank,
                    "list_id": list_id,
                    "access_count": access_count,
                    "query_access_rate": f"{access_count / num_queries:.8f}",
                }
            )


def plot_access_frequency(
    results: list[tuple[DatasetConfig, torch.Tensor, AccessSummary]],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(6.8, 1.9),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    axes = axes[0]
    kpsewhich = shutil.which("kpsewhich")
    def libertine_font(filename: str, size: float, weight: str = "normal") -> FontProperties:
        path = (subprocess.run(
            [kpsewhich, filename], capture_output=True, text=True, check=False,
        ).stdout.strip() if kpsewhich else "")
        return (FontProperties(fname=path, size=size) if path else
                FontProperties(family="Linux Libertine O", weight=weight, size=size))

    name_font = libertine_font("LinLibertine_RB.otf", 18, "bold")
    # At the paper's single-column width, these render at approximately 9 pt.
    label_font = libertine_font("LinLibertine_R.otf", 19.1)
    tick_font = libertine_font("LinLibertine_R.otf", 19.1)
    # Match the blue/red palette used by the other paper figures.
    colors = ("#00609D", "#E43D33")
    bin_width = 10

    for axis, (config, counts, summary), color in zip(axes, results, colors):
        access_counts = counts.numpy().astype(np.int64, copy=False)
        upper_edge = (int(access_counts.max()) // bin_width + 1) * bin_width
        bin_edges = np.arange(0, upper_edge + bin_width, bin_width)
        distribution, _ = np.histogram(access_counts, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        axis.plot(
            bin_centers,
            distribution,
            color=color,
            linewidth=2.2,
        )
        axis.fill_between(
            bin_centers,
            distribution,
            0,
            color=color,
            alpha=0.34,
        )
        is_hunyuan = config.key == "hunyuan" and upper_edge > 900
        axis.text(
            0.25 if is_hunyuan else 0.9,
            0.9,
            config.label,
            transform=axis.transAxes,
            ha="center" if is_hunyuan else "right",
            va="top",
            fontproperties=name_font,
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 1.0,
            },
        )
        axis.axvline(summary.mean_access_count, color="#555555", linestyle="--",
                     linewidth=1.5, zorder=4)
        axis.set_xlim(0, upper_edge)
        if is_hunyuan:
            # Compress the sparsely populated middle while preserving real x values.
            domain = np.array([0.0, 25.0, 975.0, float(upper_edge)])
            display = np.array([0.0, 0.40, 0.60, 1.0])
            axis.set_xscale("function", functions=(
                lambda values: np.interp(values, domain, display),
                lambda values: np.interp(values, display, domain),
            ))
            axis.xaxis.set_major_locator(FixedLocator([0, 25, 975, 1000]))
            axis.xaxis.set_minor_locator(FixedLocator([]))
            for xpos in (0.49, 0.51):
                axis.plot([xpos - 0.012, xpos + 0.012], [-0.027, 0.027],
                          transform=axis.transAxes, color="black", linewidth=1.8,
                          clip_on=False, zorder=6)
        axis.set_ylim(0, max(1, int(distribution.max())) * 1.08)
        axis.set_xlabel("# Visited Queries Per List", fontproperties=label_font)
        if not is_hunyuan:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if not is_hunyuan:
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.tick_params(axis="both", which="major", width=0.75, length=4.0)
        axis.tick_params(axis="both", which="minor", width=0.6, length=2.5)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_label.set_fontproperties(tick_font)
        axis.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.32)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)

    axes[0].set_ylabel("# of IVF Lists", fontproperties=label_font)

    figure.subplots_adjust(left=0.13, right=0.99, bottom=0.28, top=0.96, wspace=0.30)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "IVFList_Access_Distribution"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Use only the first N queries (default: all queries).",
    )
    parser.add_argument(
        "--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for coarse selection (default: cuda if available).",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--plot-from-summary", type=Path,
        help="Plot saved per-list CSVs using the dataset settings in this JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_from_summary is not None:
        saved = json.loads(args.plot_from_summary.read_text(encoding="utf-8"))
        summaries = {item["dataset"]: AccessSummary(**item) for item in saved["datasets"]}
        results = []
        for config in DATASETS:
            summary = summaries[config.key]
            csv_path = args.result_dir / (
                f"ivf_list_access_frequency_{config.key}_nprobe{summary.nprobe}.csv"
            )
            with csv_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            ids = [int(row["list_id"]) for row in rows]
            counts = torch.tensor([int(row["access_count"]) for row in rows])
            if (len(rows) != summary.nlist or sorted(ids) != list(range(summary.nlist))
                    or any(row["dataset"] != config.key for row in rows)
                    or int(counts.sum()) != summary.total_accesses
                    or not math.isclose(float(counts.double().mean()), summary.mean_access_count)):
                raise ValueError(f"CSV does not match summary: {csv_path}")
            results.append((config, counts, summary))
        pdf_path, png_path = plot_access_frequency(results, output_dir=args.figure_dir)
        print(f"Saved figure: {pdf_path}\nSaved figure: {png_path}")
        return
    if args.nprobe <= 0:
        raise ValueError("--nprobe must be positive")
    if args.num_queries is not None and args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    experiment_results: list[tuple[DatasetConfig, torch.Tensor, AccessSummary]] = []
    for config in DATASETS:
        dataset_root = args.data_root / config.key
        index_path = dataset_root / "GPUIndex" / config.index_filename
        query_path = dataset_root / config.query_filename
        print(f"[{config.label}] loading centroids and queries", flush=True)
        centroids, metadata = _load_index_centroids(index_path)
        queries = load_dataset(query_path)
        if args.num_queries is not None:
            queries = queries[: args.num_queries]
        if args.nprobe > centroids.shape[0]:
            raise ValueError(
                f"--nprobe={args.nprobe} exceeds {config.label} "
                f"nlist={centroids.shape[0]}"
            )

        print(
            f"[{config.label}] counting {queries.shape[0]:,} queries × "
            f"{args.nprobe} probes on {device}",
            flush=True,
        )
        counts = count_list_accesses(
            queries,
            centroids,
            nprobe=args.nprobe,
            query_batch_size=args.query_batch_size,
            device=device,
        )
        expected_accesses = queries.shape[0] * args.nprobe
        if int(counts.sum()) != expected_accesses:
            raise RuntimeError(
                f"Count invariant failed for {config.label}: expected "
                f"{expected_accesses}, got {int(counts.sum())}"
            )

        summary = summarize_accesses(
            config,
            counts,
            index_path=index_path,
            query_path=query_path,
            metadata=metadata,
            dimension=int(queries.shape[1]),
            nprobe=args.nprobe,
            num_queries=int(queries.shape[0]),
        )
        csv_path = args.result_dir / (
            f"ivf_list_access_frequency_{config.key}_nprobe{args.nprobe}.csv"
        )
        write_access_csv(csv_path, config, counts, int(queries.shape[0]))
        experiment_results.append((config, counts, summary))
        print(
            f"[{config.label}] min/mean/max={summary.min_access_count}/"
            f"{summary.mean_access_count:.2f}/{summary.max_access_count}, "
            f"top-10% share={100 * summary.top_10_percent_access_share:.2f}%, "
            f"Gini={summary.gini_coefficient:.4f}",
            flush=True,
        )
        print(f"[{config.label}] saved {csv_path}", flush=True)

    query_counts = {summary.num_queries for _, _, summary in experiment_results}
    nlists = {summary.nlist for _, _, summary in experiment_results}
    if len(query_counts) != 1 or len(nlists) != 1:
        raise ValueError("All plotted datasets must use the same query count and nlist")

    summary_path = args.result_dir / f"ivf_list_access_summary_nprobe{args.nprobe}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "definition": (
                    "One access is counted when an IVF list appears in a query's "
                    "coarse nprobe selection."
                ),
                "device": str(device),
                "datasets": [asdict(summary) for _, _, summary in experiment_results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pdf_path, png_path = plot_access_frequency(
        experiment_results,
        output_dir=args.figure_dir,
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
