"""Plot AMD MI300X IVF-Flat or IVF-PQ Recall-QPS results in the MPS layout.

Rows are GloVe and Hunyuan; columns are QBatch=32, 128, and 512.  Each
panel compares completed hipVS, TensorVS (Tensor), and optional Triton CSVs.

Run on the AMD server:
    /root/tensorvs-venv/bin/python plot/plot_amd_ivf_compare.py --kind ivfflat
    /root/tensorvs-venv/bin/python plot/plot_amd_ivf_compare.py --kind ivfpq
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (("glove1.2m", "GloVe", "M32_nbits8_train512"),
            ("hunyuan", "Hunyuan", "M384_nbits8_train256"))
QBATCHES = (32, 128, 512)
BASE_METHODS = ("hipVS", "TensorVS (Tensor)")
TRITON_METHOD = "TensorVS (Triton)"
FIELDS = ("nprobes", "q_batch", "K", "qps", "recall",
          "visited_vectors_per_query", "peak_gpu_memory")
STYLES = {
    "hipVS": dict(color="#e43d33", marker="o", linestyle="-",
                  markerfacecolor="#e43d33"),
    "TensorVS (Tensor)": dict(color="#f5ba03", marker="o", linestyle="-",
                              markerfacecolor="#f5ba03"),
    "TensorVS (Triton)": dict(color="#00609d", marker="s", linestyle="-",
                              markerfacecolor="#00609d"),
}


def result_path(result_dir: Path, method: str, kind: str,
                dataset: str, pq_config: str) -> Path:
    prefix = {"hipVS": "hipVS", "TensorVS (Tensor)": "TensorVS_Tensor",
              "TensorVS (Triton)": "TensorVS_Triton"}[method]
    suffix = f"_{pq_config}" if kind == "ivfpq" else ""
    return result_dir / f"{prefix}_{kind.upper()}_{dataset}_nlist1024{suffix}.csv"


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


def plot(result_dir: Path, output_dir: Path, kind: str) -> tuple[Path, Path]:
    methods = BASE_METHODS + (TRITON_METHOD,)
    curves = {}
    for dataset, _label, pq_config in DATASETS:
        curves[dataset] = {
            method: read_curves(result_path(result_dir, method, kind, dataset, pq_config))
            for method in methods
        }
        for batch in QBATCHES:
            probes = [set(point[0] for point in curves[dataset][method][batch])
                      for method in methods]
            if any(probe_set != probes[0] for probe_set in probes[1:]):
                raise ValueError(f"Methods have different probes for {dataset}, QBatch={batch}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13.5, "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(2, 3, figsize=(9.8, 5.25), squeeze=False)
    figure.subplots_adjust(left=.085, right=.975, top=.78, bottom=.12,
                           wspace=.20, hspace=.48 if kind == "ivfpq" else .26)
    handles = {}

    for row_index, (dataset, label, _pq_config) in enumerate(DATASETS):
        # IVF-Flat Tensor can be fastest at high recall. A Pareto filter then
        # removes nearly every lower-nprobe measurement and hides its curve.
        # Show the complete measured nprobe sweep for each method instead.
        frontiers = {
            (method, batch): (sorted(curves[dataset][method][batch])
                              if kind == "ivfflat" else
                              performance_frontier(curves[dataset][method][batch]))
            for method in methods for batch in QBATCHES
        }
        all_points = [point for points in frontiers.values() for point in points]
        if kind == "ivfflat":
            xlim, recall_format = (.85, 1.005), "%.2f"
        else:
            recall_low = min(point[1] for point in all_points)
            recall_high = max(point[1] for point in all_points)
            recall_span = max(recall_high - recall_low, .01)
            recall_format = "%.3f" if recall_span < .05 else "%.2f"
            xlim = (max(0, recall_low - .05 * recall_span),
                    min(1.005, recall_high + .05 * recall_span))

        for column, batch in enumerate(QBATCHES):
            axis = axes[row_index][column]
            visible_qps = [point[2] for method in methods
                           for point in frontiers[(method, batch)]
                           if xlim[0] <= point[1] <= xlim[1]]
            if not visible_qps:
                raise ValueError(f"No {dataset} results within Recall range {xlim}")
            low, high = min(visible_qps), max(visible_qps)
            ylim = (10 ** (math.log10(low) - .12),
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
            axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
            axis.tick_params(axis="both", labelsize=14)
            axis.tick_params(axis="x",
                             labelbottom=kind == "ivfpq" or row_index == len(DATASETS) - 1,
                             labeltop=False, top=False)
            if row_index == len(DATASETS) - 1:
                axis.set_xlabel("Recall", fontsize=16)
            if column == 0:
                axis.set_ylabel("QPS", fontsize=16)
            axis.set_title(f"{label} (QBatch = {batch})", fontsize=12.5,
                           fontweight="bold", pad=5)

    figure.legend([handles[method] for method in methods], list(methods),
                  loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=len(methods),
                  frameon=True, fontsize=14, handlelength=2.5,
                  columnspacing=1.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{kind.upper()}_AMD_QBatch32_128_512_K10"
    pdf_path, png_path = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    figure.savefig(pdf_path, dpi=300)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("ivfflat", "ivfpq"), required=True)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot" / "Figures")
    args = parser.parse_args()
    plot(args.result_dir, args.output_dir, args.kind)


if __name__ == "__main__":
    main()
