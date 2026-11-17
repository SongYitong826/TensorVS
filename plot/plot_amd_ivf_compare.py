"""Plot AMD MI300X IVF-Flat or IVF-PQ Recall-QPS results in the MPS layout.

Rows are GloVe and Hunyuan; columns are QBatch=32, 128, and 512.  Each
panel compares hipVS, TensorVS (Tensor/Triton), and Faiss ROCm CSVs.
Faiss ROCm is omitted for Hunyuan IVF-PQ, which it does not support.
Use --q-batch to plot one batch with the two datasets side by side.
On the local paper workspace, generated PDFs are also copied to
~/Desktop/TensorVS_SIGMOD27/expFig when that directory exists.

Run on the AMD server:
    /root/tensorvs-venv/bin/python plot/plot_amd_ivf_compare.py --kind ivfflat
    /root/tensorvs-venv/bin/python plot/plot_amd_ivf_compare.py --kind ivfpq
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURE_DIR = Path.home() / "Desktop" / "TensorVS_SIGMOD27" / "expFig"
DATASETS = (("glove1.2m", "GloVe", "M32_nbits8_train512"),
            ("hunyuan", "Hunyuan", "M384_nbits8_train256"))
QBATCHES = (32, 128, 512)
BASE_METHODS = ("hipVS", "TensorVS (Tensor)")
TRITON_METHOD = "TensorVS (Triton)"
FAISS_METHOD = "Faiss ROCm"
METHODS = ("hipVS", FAISS_METHOD, "TensorVS (Tensor)", TRITON_METHOD)
FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
          "visited_vectors_per_query", "peak_gpu_memory")
STYLES = {
    FAISS_METHOD: dict(color="#2ca02c", marker="^", linestyle="-",
                      markerfacecolor="#2ca02c"),
    "hipVS": dict(color="#e43d33", marker="o", linestyle="-",
                  markerfacecolor="#e43d33"),
    "TensorVS (Tensor)": dict(color="#f5ba03", marker="o", linestyle="-",
                              markerfacecolor="#f5ba03"),
    "TensorVS (Triton)": dict(color="#00609d", marker="s", linestyle="-",
                              markerfacecolor="#00609d"),
}


def methods_for(dataset: str, kind: str) -> tuple[str, ...]:
    return tuple(method for method in METHODS
                 if not (method == FAISS_METHOD and dataset == "hunyuan" and kind == "ivfpq"))


def result_path(result_dir: Path, method: str, kind: str,
                dataset: str, pq_config: str) -> Path:
    prefix = {"hipVS": "hipVS", "TensorVS (Tensor)": "TensorVS_Tensor",
              "TensorVS (Triton)": "TensorVS_Triton", FAISS_METHOD: "FaissROCm"}[method]
    if method == FAISS_METHOD and kind == "ivfpq":
        if dataset != "glove1.2m":
            raise ValueError(f"Faiss ROCm IVF-PQ is not supported for {dataset}")
        # The Faiss run uses its own train label and zero-padded dimension.
        pq_config = "M32_nbits8_train256_pad224"
    suffix = f"_{pq_config}" if kind == "ivfpq" else ""
    stem = f"{prefix}_{kind.upper()}_{dataset}_nlist1024{suffix}"
    marked = result_dir / f"AMD_{stem}.csv"
    # Retain compatibility with historical local checkouts not yet renamed.
    legacy = result_dir / f"{stem}.csv"
    return marked if marked.exists() or not legacy.exists() else legacy


def read_curves(path: Path) -> dict[int, list[tuple[int, float, float]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Unexpected CSV columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    with path.with_suffix(".json").open(encoding="utf-8") as source:
        metadata = json.load(source)
    if metadata.get("status") != "complete" or metadata.get("completed_rows") != len(rows):
        raise ValueError(f"Incomplete benchmark: {path}")
    curves: dict[int, list[tuple[int, float, float]]] = {batch: [] for batch in QBATCHES}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        batch, probe, topk = int(row["q_batch"]), int(row["nprobes"]), int(row["K"])
        key = batch, probe
        if batch not in curves or topk != 10 or key in seen:
            raise ValueError(f"Unexpected or repeated configuration {key} in {path}")
        seen.add(key)
        qps, recall = float(row["qps"]), float(row["recall"])
        visited, peak_mib = (float(row["visited_vectors_per_query"]),
                             float(row["peak_gpu_memory"]))
        if not (math.isfinite(qps) and qps > 0 and math.isfinite(recall) and
                0 <= recall <= 1 and math.isfinite(visited) and visited >= 0 and
                math.isfinite(peak_mib) and peak_mib > 0):
            raise ValueError(f"Invalid metric for {key} in {path}")
        curves[batch].append((probe, recall, qps))
    expected = {(batch, probe) for batch in metadata["config"]["q_batches"]
                for probe in metadata["config"]["nprobes"]}
    if seen != expected or not all(curves.values()):
        raise ValueError(f"Missing or unexpected benchmark rows in {path}")
    for points in curves.values():
        points.sort()
    return curves


def performance_frontier(points: list[tuple[int, float, float]]):
    """Use the same non-dominated Recall-QPS line as the MPS reference."""
    frontier: list[tuple[int, float, float]] = []
    fastest = -math.inf
    for point in sorted(points, key=lambda value: (-round(value[1], 12), -value[2])):
        if point[2] > fastest:
            frontier.append(point)
            fastest = point[2]
    return list(reversed(frontier))


def plot(result_dir: Path, output_dir: Path, kind: str,
         q_batch: int | None = None) -> tuple[Path, Path]:
    batches = (q_batch,) if q_batch is not None else QBATCHES
    single_batch = q_batch is not None
    curves = {}
    for dataset, _label, pq_config in DATASETS:
        methods = methods_for(dataset, kind)
        curves[dataset] = {
            method: read_curves(result_path(result_dir, method, kind, dataset, pq_config))
            for method in methods
        }
        for batch in batches:
            probes = [set(point[0] for point in curves[dataset][method][batch])
                      for method in methods]
            if any(probe_set != probes[0] for probe_set in probes[1:]):
                raise ValueError(f"Methods have different probes for {dataset}, QBatch={batch}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13.5, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    mps_layout = not single_batch and kind == "ivfflat"
    if single_batch:
        figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), squeeze=False)
        figure.subplots_adjust(left=.12, right=.97, top=.79, bottom=.23, wspace=.38)
    else:
        # Match the MPS IVF-Flat layout.
        figure, axes = plt.subplots(2, 3, figsize=(9.8, 4.34), squeeze=False)
        if mps_layout:
            figure.subplots_adjust(left=.085, right=.975, top=.79, bottom=.145,
                                   wspace=.30, hspace=.18)
        else:
            figure.subplots_adjust(left=.085, right=.99, top=.79, bottom=.145,
                                   wspace=.18, hspace=.28)
    legend_bottom = .833 if mps_layout else .825 if single_batch else .802
    handles = {}

    for row_index, (dataset, label, _pq_config) in enumerate(DATASETS):
        methods = methods_for(dataset, kind)
        # IVF-Flat Tensor can be fastest at high recall. A Pareto filter then
        # removes nearly every lower-nprobe measurement and hides its curve.
        # Show the complete measured nprobe sweep for each method instead.
        frontiers = {
            (method, batch): (sorted(curves[dataset][method][batch])
                              if kind == "ivfflat" else
                              performance_frontier(curves[dataset][method][batch]))
            for method in methods for batch in batches
        }
        all_points = [point for points in frontiers.values() for point in points]
        if kind == "ivfflat":
            xlim, recall_format = (.85, 1.005), "%.2f"
        else:
            recall_low = min(point[1] for point in all_points)
            recall_high = max(point[1] for point in all_points)
            recall_span = max(recall_high - recall_low, .01)
            recall_format = "%.2f"
            xlim = (max(0, recall_low - .05 * recall_span),
                    min(1.005, recall_high + .05 * recall_span))

        for column, batch in enumerate(batches):
            axis = axes[0][row_index] if single_batch else axes[row_index][column]
            visible_qps = [point[2] for method in methods
                           for point in frontiers[(method, batch)]
                           if xlim[0] <= point[1] <= xlim[1]]
            if not visible_qps:
                raise ValueError(f"No {dataset} results within Recall range {xlim}")
            low, high = min(visible_qps), max(visible_qps)
            # Reserve space below the curves for the panel annotation.
            ylim = (10 ** (math.log10(low) - .75),
                    10 ** (math.log10(high) + .12))
            for method in methods:
                points = frontiers[(method, batch)]
                (line,) = axis.plot(
                    [point[1] for point in points], [point[2] for point in points],
                    label=method, linewidth=1.7, markersize=7.0,
                    markeredgecolor=STYLES[method]["color"],
                    markeredgewidth=1.0, **STYLES[method],
                )
                handles.setdefault(method, line)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.yaxis.set_minor_locator(LogLocator(base=10,
                subs=tuple(range(2, 10)), numticks=12))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.xaxis.set_major_formatter(FormatStrFormatter(recall_format))
            if single_batch:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
            axis.tick_params(axis="both", labelsize=14)
            axis.tick_params(axis="x",
                             labelbottom=not mps_layout or row_index == len(DATASETS) - 1,
                             labeltop=False, top=False)
            if single_batch or row_index == len(DATASETS) - 1:
                xlabel = f"Recall (QBatch = {batch})" if mps_layout else "Recall"
                axis.set_xlabel(xlabel, fontsize=16)
            if column == 0 and (not single_batch or row_index == 0):
                axis.set_ylabel("QPS", fontsize=16)
            panel_label = label if single_batch or mps_layout else f"{label} (QBatch = {batch})"
            axis.text(.04, .05, panel_label, transform=axis.transAxes,
                      fontsize=15 if mps_layout else 14 if single_batch else 11.5,
                      fontweight="bold", ha="left", va="bottom")

    legend_center = (.5 if single_batch or mps_layout else
                     (figure.subplotpars.left + figure.subplotpars.right) / 2)
    figure.legend([handles[method] for method in METHODS], list(METHODS),
                  loc="lower center", bbox_to_anchor=(legend_center, legend_bottom),
                  borderaxespad=0, ncol=len(METHODS),
                  frameon=True, fontsize=16 if mps_layout else 12 if single_batch else 14,
                  handlelength=2 if mps_layout else 1.5 if single_batch else 2.5,
                  handletextpad=.5 if single_batch else .8,
                  columnspacing=.9 if single_batch or mps_layout else 1.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_suffix = "_".join(str(batch) for batch in batches)
    stem = f"{kind.upper()}_AMD_QBatch{batch_suffix}_K10"
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    figure.savefig(pdf_path, dpi=300)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    if PAPER_FIGURE_DIR.is_dir():
        paper_pdf = PAPER_FIGURE_DIR / pdf_path.name
        if paper_pdf.resolve() != pdf_path.resolve():
            shutil.copy2(pdf_path, paper_pdf)
            print(f"Synced {paper_pdf}")
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("ivfflat", "ivfpq"), required=True)
    parser.add_argument("--q-batch", type=int, choices=QBATCHES,
                        help="Plot one batch in a 1x2 dataset layout.")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot" / "Figures")
    args = parser.parse_args()
    plot(args.result_dir, args.output_dir, args.kind, q_batch=args.q_batch)


if __name__ == "__main__":
    main()
