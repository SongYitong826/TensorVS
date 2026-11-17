"""Plot Ascend NPU IVF-Flat / IVF-PQ Recall-QPS results in the AMD/MPS layout.

Rows are GloVe and Hunyuan; columns are QBatch=32, 128, and 512.  Each panel
compares TensorVS (Tensor, torch_npu) against Faiss on Ascend.
Selecting --q-batch produces one row with GloVe and Hunyuan side by side.

Faiss on Ascend does not implement IVF-Flat, and it does not implement IVF-PQ
for Hunyuan, so those panels use separately identified Faiss CPU measurements.
The Hunyuan IVF-PQ fallback is measured on the local Apple Silicon Mac.

Run from the TensorVS project root:
    python plot/plot_ascend_ivf_compare.py --kind ivfflat
    python plot/plot_ascend_ivf_compare.py --kind ivfpq
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (FormatStrFormatter, LogLocator, MultipleLocator,
                               NullFormatter)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench._mps_results import read_rows as read_mps_rows

PAPER_FIGURE_DIR = Path("/Users/yitongsong/Desktop/TensorVS_SIGMOD27/expFig")
DATASETS = (("glove1.2m", "GloVe", "M32_nbits8_train512"),
            ("hunyuan", "Hunyuan", "M384_nbits8_train256"))
QBATCHES = (32, 128, 512)
FAISS_METHOD = "Faiss Ascend"
TENSOR_METHOD = "TensorVS (Tensor)"
CPU_METHOD = "FAISS CPU (fallback)"
MAC_CPU_LABEL = "Faiss CPU (Mac fallback)"
MAC_CPU_PREFIX = "MPS_FaissCPU"
METHODS = (FAISS_METHOD, TENSOR_METHOD)
LEGEND_METHODS = (FAISS_METHOD, CPU_METHOD, TENSOR_METHOD)
FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
          "visited_vectors_per_query", "peak_gpu_memory")
STYLES = {
    FAISS_METHOD: dict(color="#e43d33", marker="o", linestyle="-",
                       markerfacecolor="#e43d33"),
    TENSOR_METHOD: dict(color="#00609d", marker="s", linestyle="-",
                        markerfacecolor="#00609d"),
    CPU_METHOD: dict(color="#2ca02c", marker="^", linestyle="-",
                     markerfacecolor="#2ca02c"),
}


def result_path(result_dir: Path, method: str, kind: str,
                dataset: str, pq_config: str) -> Path:
    """Locate the CSV for one panel.

    Native NPU runs are stored as ``Ascend_<prefix>_...``.  The combos Faiss
    cannot run on the NPU use CPU measurements instead: GloVe IVF-Flat was
    recorded under the Ascend name with ``device=cpu``. Hunyuan IVF-Flat uses
    the server CPU result, and Hunyuan IVF-PQ uses the local Mac CPU result.
    """
    if method == TENSOR_METHOD:
        prefix = "Ascend_TensorVS_Tensor"
    elif dataset == "glove1.2m":
        prefix = "Ascend_Faiss"
    elif kind == "ivfflat":
        prefix = "CPU_Faiss"
    else:
        prefix = MAC_CPU_PREFIX
    suffix = f"_{pq_config}" if kind == "ivfpq" else ""
    return result_dir / f"{prefix}_{kind.upper()}_{dataset}_nlist1024{suffix}.csv"


def read_curves(path: Path) -> dict[int, list[tuple[int, float, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {path}")
    with path.with_suffix(".json").open(encoding="utf-8") as source:
        metadata = json.load(source)
    if path.name.startswith(f"{MAC_CPU_PREFIX}_"):
        rows = read_mps_rows(path)
        if metadata.get("host_family") != "Apple Silicon" or any(
            row.get("method") != "Faiss CPU" or row.get("execution_device") != "cpu"
            or row.get("dataset") != "hunyuan" or row.get("index_type") != "ivfpq"
            for row in rows
        ):
            raise ValueError(f"Expected local Hunyuan Faiss CPU IVF-PQ measurements: {path}")
    else:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"Unexpected CSV columns in {path}: {reader.fieldnames}")
            rows = list(reader)
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
        if not (math.isfinite(qps) and qps > 0 and math.isfinite(recall) and
                0 <= recall <= 1):
            raise ValueError(f"Invalid metric for {key} in {path}")
        curves[batch].append((probe, recall, qps))
    if not all(curves.values()):
        raise ValueError(f"Missing benchmark rows in {path}")
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


def recall_tick_step(xlim: tuple[float, float], max_ticks: int = 6) -> float:
    """Pick a round Recall step so tick labels stay evenly spaced.

    AutoLocator picks multiples of the raw span (e.g. 0.375), which then
    collapses to uneven-looking 0.35/0.38/0.40 labels under a %.2f format.
    """
    span = xlim[1] - xlim[0]
    for step in (.01, .02, .025, .05, .1, .2):
        if span / step <= max_ticks:
            return step
    return .2


def plot(result_dir: Path, output_dir: Path, kind: str,
         q_batch: int | None = None, *, sync_paper: bool = True) -> tuple[Path, Path]:
    batches = (q_batch,) if q_batch is not None else QBATCHES
    single_batch = q_batch is not None
    curves = {}
    execution_methods = {}
    for dataset, _label, pq_config in DATASETS:
        curves[dataset] = {}
        for method in METHODS:
            path = result_path(result_dir, method, kind, dataset, pq_config)
            curves[dataset][method] = read_curves(path)
            metadata = json.loads(path.with_suffix(".json").read_text())
            cpu = (path.name.startswith("CPU_") or metadata.get("device") == "cpu" or
                   metadata.get("config", {}).get("execution_device") == "cpu" or
                   metadata.get("backend") == "faiss_cpu")
            execution_methods[dataset, method] = CPU_METHOD if method == FAISS_METHOD and cpu else method

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13.5, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    if single_batch:
        compact_flat = kind == "ivfflat"
        height = 2.33 if compact_flat else 2.5
        figure, axes = plt.subplots(1, 2, figsize=(6.4, height), squeeze=False)
        figure.subplots_adjust(left=.123 if compact_flat else .115, right=.97,
                               top=.79, bottom=.25 if compact_flat else .23, wspace=.34)
    else:
        # Match the AMD/MPS canvas.
        figure, axes = plt.subplots(2, 3, figsize=(9.8, 4.34), squeeze=False)
        figure.subplots_adjust(left=.085, right=.99 if kind == "ivfflat" else .98,
                               top=.79, bottom=.145,
                               wspace=.18, hspace=.05 if kind == "ivfflat" else .28)
    legend_bottom = .825 if single_batch else .802
    handles = {}

    # IVF-Flat Tensor can be fastest at high recall. A Pareto filter then
    # removes nearly every lower-nprobe measurement and hides its curve.
    # Show the complete measured nprobe sweep for each method instead.
    frontiers = {
        (dataset, method, batch): (
            sorted(curves[dataset][method][batch]) if kind == "ivfflat" else
            performance_frontier(curves[dataset][method][batch]))
        for dataset, _label, _pq in DATASETS
        for method in METHODS for batch in batches
    }

    def recall_range(datasets) -> tuple[float, float]:
        points = [point for (dataset, _method, _batch), series in frontiers.items()
                  if dataset in datasets for point in series]
        low = min(point[1] for point in points)
        high = max(point[1] for point in points)
        pad = .03 if kind == "ivfflat" else .05
        span = max(high - low, .01)
        # A hidden top-row tick label must not hide a different scale, so
        # IVF-Flat shares one Recall range across rows (as the AMD figure does).
        return (max(0., low - pad * span), min(1.005, high + pad * span))

    shared_xlim = (.85, 1.005) if kind == "ivfflat" else None

    for row_index, (dataset, label, _pq_config) in enumerate(DATASETS):
        xlim = shared_xlim or recall_range({dataset})
        recall_format = "%.2f"

        for column, batch in enumerate(batches):
            axis = axes[0][row_index] if single_batch else axes[row_index][column]
            visible_qps = [point[2] for method in METHODS
                           for point in frontiers[(dataset, method, batch)]
                           if xlim[0] <= point[1] <= xlim[1]]
            if not visible_qps:
                raise ValueError(f"No {dataset} results within Recall range {xlim}")
            low, high = min(visible_qps), max(visible_qps)
            # Reserve space below the curves for the panel annotation, and
            # enough headroom on top that the largest markers stay unclipped.
            ylim = (10 ** (math.log10(low) - .75),
                    10 ** (math.log10(high) + .18))
            for method in METHODS:
                points = frontiers[(dataset, method, batch)]
                display_method = execution_methods[dataset, method]
                (line,) = axis.plot(
                    [point[1] for point in points], [point[2] for point in points],
                    label=display_method, linewidth=1.7, markersize=7.0,
                    markeredgecolor=STYLES[display_method]["color"],
                    markeredgewidth=1.0, **STYLES[display_method],
                )
                handles.setdefault(display_method, line)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.yaxis.set_minor_locator(LogLocator(base=10,
                subs=tuple(range(2, 10)), numticks=12))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.xaxis.set_major_formatter(FormatStrFormatter(recall_format))
            axis.xaxis.set_major_locator(MultipleLocator(recall_tick_step(xlim)))
            axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
            axis.tick_params(axis="both", labelsize=14)
            axis.tick_params(axis="x",
                             labelbottom=single_batch or kind == "ivfpq" or row_index == len(DATASETS) - 1,
                             labeltop=False, top=False)
            if single_batch or row_index == len(DATASETS) - 1:
                axis.set_xlabel("Recall", fontsize=16)
            if column == 0 and (not single_batch or row_index == 0):
                axis.set_ylabel("QPS", fontsize=16)
            panel_label = label if single_batch else f"{label} (QBatch = {batch})"
            axis.text(.04, .05, panel_label, transform=axis.transAxes,
                      fontsize=14 if single_batch else 11.5,
                      fontweight="bold", ha="left", va="bottom")

    legend_methods = [method for method in LEGEND_METHODS if method in handles]
    fallback_label = "Faiss CPU(fallback)" if single_batch else MAC_CPU_LABEL
    legend_labels = [fallback_label if kind == "ivfpq" and method == CPU_METHOD else method
                     for method in legend_methods]
    compact_legend = single_batch and len(legend_methods) == 3
    legend_center = (.5 if compact_legend else
                     (figure.subplotpars.left + figure.subplotpars.right) / 2)
    figure.legend([handles[method] for method in legend_methods], legend_labels,
                  loc="lower center", bbox_to_anchor=(legend_center, legend_bottom),
                  borderaxespad=0, ncol=len(legend_methods),
                  frameon=True, fontsize=12.5 if compact_legend else 14,
                  handlelength=1.8 if compact_legend else 2.5,
                  columnspacing=1 if compact_legend else 1.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_suffix = "_".join(str(batch) for batch in batches)
    stem = f"{kind.upper()}_Ascend_QBatch{batch_suffix}_K10"
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    figure.savefig(pdf_path, dpi=300)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    paper_pdf = PAPER_FIGURE_DIR / pdf_path.name
    if sync_paper and PAPER_FIGURE_DIR.is_dir() and paper_pdf.resolve() != pdf_path.resolve():
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
