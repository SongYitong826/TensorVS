"""Plot the GloVe/Hunyuan Apple-MPS Recall--QPS experiment.

Each index type is written as a separate 2x3 figure. Rows are datasets
(GloVe and Hunyuan), columns are QBatch=32/128/512, and K is fixed at 10.
The input CSVs are produced by ``bench/compare_mps_faiss.py`` and must include
sampled peak Metal-driver and process-RSS memory for every measured point.
CPU execution (Faiss IVF-PQ) may report zero Metal allocation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench._mps_results import comparison_rows

DATASETS = (("glove1.2m", "GloVe"), ("hunyuan", "Hunyuan"))
QBATCHES = (32, 128, 512)
METHODS = ("TensorVS", "Faiss Metal")
STYLES = {
    "TensorVS": dict(
        color="#00609d", marker="s", linestyle="-", markerfacecolor="#00609d"
    ),
    "Faiss Metal": dict(
        color="#e43d33", marker="o", linestyle="-", markerfacecolor="#e43d33"
    ),
}
LEGEND_ORDER = ("Faiss Metal", "TensorVS")


def read_curves(path: Path, dataset: str, kind: str):
    rows = comparison_rows(path, dataset, kind)
    curves: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        if row["dataset"] != dataset or row["index_type"] != kind:
            raise ValueError(f"Misidentified row in {path}")
        method = row["method"]
        batch = int(row["q_batch"])
        topk = int(row["K"])
        probe = int(row["nprobes"])
        key = method, batch, probe
        if method not in METHODS or batch not in QBATCHES or topk != 10 or key in seen:
            raise ValueError(f"Unexpected or duplicate configuration in {path}: {key}, K={topk}")
        seen.add(key)
        qps, recall = float(row["qps"]), float(row["recall"])
        peak_metal = float(row["peak_gpu_memory"])
        peak_rss = float(row["peak_rss_memory"])
        if not (math.isfinite(qps) and qps > 0 and 0 <= recall <= 1):
            raise ValueError(f"Invalid performance values in {path}: {key}")
        cpu = row.get("execution_device") == "cpu"
        if not (math.isfinite(peak_metal) and (peak_metal >= 0 if cpu else peak_metal > 0) and
                math.isfinite(peak_rss) and peak_rss > 0):
            raise ValueError(f"Missing peak-memory measurement in {path}: {key}")
        curves.setdefault((method, batch), []).append((probe, recall, qps))
    required_methods = ("TensorVS",) if dataset == "hunyuan" and kind == "ivfpq" and not any(
        method == "Faiss Metal" for method, _ in curves) else METHODS
    for method in required_methods:
        for batch in QBATCHES:
            if not curves.get((method, batch)):
                raise ValueError(f"Missing {method}, QBatch={batch} in {path}")
    for points in curves.values():
        points.sort()
    return curves, rows


def performance_frontier(points: list[tuple[int, float, float]]):
    frontier: list[tuple[int, float, float]] = []
    fastest = -math.inf
    for point in sorted(points, key=lambda value: (-round(value[1], 12), -value[2])):
        if point[2] > fastest:
            frontier.append(point)
            fastest = point[2]
    return list(reversed(frontier))


def plot_kind(kind: str, result_dir: Path, output_dir: Path):
    loaded = {
        dataset: read_curves(result_dir, dataset, kind)
        for dataset, _ in DATASETS
    }
    curves = {dataset: values[0] for dataset, values in loaded.items()}
    rows = [row for values in loaded.values() for row in values[1]]
    display_kind = "IVF-Flat" if kind == "ivfflat" else "IVF-PQ"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13.5, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(2, 3, figsize=(9.8, 4.34), squeeze=False)
    figure.subplots_adjust(
        left=.085,
        right=.99,
        top=.79,
        bottom=.145,
        wspace=.18,
        hspace=.05,
    )
    handles: dict[str, object] = {}

    for row_index, (dataset, label) in enumerate(DATASETS):
        row_frontiers = {
            key: performance_frontier(points) for key, points in curves[dataset].items()
        }
        all_points = [point for points in row_frontiers.values() for point in points]
        recall_low = min(point[1] for point in all_points)
        recall_high = max(point[1] for point in all_points)
        recall_span = max(recall_high - recall_low, .01)
        recall_formatter = "%.3f" if recall_span < .05 else "%.2f"
        xlim = (max(0, recall_low - .05 * recall_span),
                min(1.005, recall_high + .05 * recall_span))
        qps_low = min(point[2] for point in all_points)
        qps_high = max(point[2] for point in all_points)
        ylim = (10 ** (math.log10(qps_low) - .08), 10 ** (math.log10(qps_high) + .08))

        for column, batch in enumerate(QBATCHES):
            axis = axes[row_index][column]
            for method in METHODS:
                points = row_frontiers.get((method, batch))
                if not points:
                    continue
                (line,) = axis.plot(
                    [point[1] for point in points], [point[2] for point in points],
                    label=method,
                    linewidth=1.7,
                    markersize=7.0,
                    markeredgecolor=STYLES[method]["color"],
                    markeredgewidth=1.0,
                    **STYLES[method],
                )
                handles.setdefault(method, line)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.xaxis.set_major_formatter(FormatStrFormatter(recall_formatter))
            axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
            axis.tick_params(axis="both", labelsize=14)
            axis.tick_params(axis="x", labelbottom=row_index == len(DATASETS) - 1)
            if row_index == len(DATASETS) - 1:
                axis.set_xlabel("Recall", fontsize=16)
            if column == 0:
                axis.set_ylabel("QPS", fontsize=16)
            axis.text(
                .04,
                .06,
                label,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=15.5,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78},
            )
            axis.text(
                .96,
                .94,
                f"QBatch = {batch}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=14,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78},
            )

    figure.legend(
        [handles[method] for method in LEGEND_ORDER if method in handles],
        [method for method in LEGEND_ORDER if method in handles],
        loc="upper center",
        bbox_to_anchor=(.5, .955),
        ncol=2,
        frameon=True,
        fontsize=16.5,
        handlelength=2.8,
        columnspacing=2.2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{'IVFFlat' if kind == 'ivfflat' else 'IVFPQ'}_MPS_QBatch32_128_512_K10"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    # Keep both MPS figures on the same fixed canvas so they align exactly
    # when used at the same column width in the paper.
    figure.savefig(pdf_path, dpi=300)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)

    peak_metal = max(float(row["peak_gpu_memory"]) for row in rows)
    peak_rss = max(float(row["peak_rss_memory"]) for row in rows)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"{display_kind}: maximum sampled Peak Metal={peak_metal:.1f} MiB, "
          f"Peak RSS={peak_rss:.1f} MiB")
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot" / "Figures")
    args = parser.parse_args()
    for kind in ("ivfflat", "ivfpq"):
        plot_kind(kind, args.result_dir, args.output_dir)


if __name__ == "__main__":
    main()
