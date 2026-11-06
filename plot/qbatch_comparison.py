"""Plot three QBatch panels from the final comparison CSVs; PNG/PDF only."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, MultipleLocator, NullFormatter, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench._mps_results import comparison_rows

BATCHES = (128, 256, 512)
METHODS = ("Faiss Metal", "TensorVS")


def read_results(path, dataset, kind):
    rows = comparison_rows(path, dataset, kind, archive=True)
    series, keys = {}, set()
    for row in rows:
        batch, probe, k = (int(row[field]) for field in ("q_batch", "nprobes", "K"))
        key = row["method"], batch, probe
        if key in keys or row["dataset"] != dataset or row["index_type"] != kind:
            raise ValueError(f"Duplicate or misidentified measurements in {path}")
        keys.add(key)
        if k != 10 or batch not in BATCHES or row["method"] not in METHODS:
            raise ValueError(f"Unexpected plotted configuration: {key}")
        qps, recall = float(row["qps"]), float(row["recall"])
        if not math.isfinite(qps) or qps <= 0 or not math.isfinite(recall) or not 0 <= recall <= 1:
            raise ValueError(f"Invalid plotted point: {key}")
        query_count = int(row["query_count"])
        repetitions = int(row["repetitions"])
        if query_count <= 0 or repetitions < 3 or repetitions % 2 != 1:
            raise ValueError("Comparison requires positive query counts and at least three odd repetitions")
        if row["warmup_scope"] != "all-batches" or int(row["warmup_iterations"]) != 2:
            raise ValueError("Comparison requires full-query warmup")
        samples = [float(row[f"sample{i}_seconds"]) for i in range(1, repetitions + 1)]
        if not all(math.isfinite(value) and value > 0 for value in samples):
            raise ValueError("Invalid timing samples")
        elapsed = statistics.median(samples)
        if not math.isclose(qps, query_count / elapsed, rel_tol=1e-9):
            raise ValueError("QPS does not match saved timing samples")
        expected_device = "mps" if row["method"] == "TensorVS" else "metal" if kind == "ivfflat" else "cpu"
        if row["execution_device"] != expected_device:
            raise ValueError("Incorrect execution-device label")
        series.setdefault((row["method"], batch), []).append((probe, recall, qps))
    configuration_fields = ("index_path", "index_size_bytes", "index_mtime_ns", "query_path",
                            "groundtruth_path", "query_count", "nlist", "dim", "M", "nbits", "pq_working_dim")
    for field in configuration_fields:
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"Methods/batches use different {field}")
    configurations = []
    for method in METHODS:
        for batch in BATCHES:
            points = sorted(series.get((method, batch), []))
            if not points:
                raise ValueError(f"Missing {method}, QBatch={batch}")
            series[method, batch] = points
            configurations.append(tuple(p[0] for p in points))
    expected_probes = ((50, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,
                        280, 300, 320, 340, 360, 380, 400) if dataset.startswith("glove") else
                       (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150))
    if any(probes != expected_probes for probes in configurations):
        raise ValueError("Missing or inconsistent nprobe settings")
    return rows, series


def performance_frontier(points):
    """Keep measured points not dominated in both Recall and throughput."""
    frontier, fastest = [], -math.inf
    # Recall is discrete; ignore summation roundoff when comparing equal recall.
    for point in sorted(points, key=lambda p: (-round(p[1], 12), -p[2])):
        if point[2] > fastest:
            frontier.append(point)
            fastest = point[2]
    return list(reversed(frontier))


def make_figure(dataset, kind, results_dir, output_dir, curve_mode="frontier"):
    rows, series = read_results(results_dir, dataset, kind)
    flat = kind == "ivfflat"
    label = "SIFT1M" if dataset.startswith("sift") else "GloVe1.2M"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.4), sharex=True, sharey=True)
    figure.subplots_adjust(left=.066, right=.985, top=.865, bottom=.16, wspace=.13)
    styles = {"Faiss Metal": dict(color="#2474B5", marker="o", linestyle="-"),
              "TensorVS": dict(color="#D66A28", marker="s", linestyle="--", markerfacecolor="white")}
    labels = {method: method for method in METHODS}
    drawn_series = {key: performance_frontier(points) if curve_mode == "frontier" else points
                    for key, points in series.items()}
    all_points = [p for points in drawn_series.values() for p in points]
    lo, hi = min(p[1] for p in all_points), max(p[1] for p in all_points)
    padding = max(.008, (hi - lo) * .045)
    xmin, xmax = (.9, 1.005) if flat else (max(0, lo - padding), min(1.009, hi + padding))
    visible_rates = [p[2] for p in all_points if xmin <= p[1] <= xmax]
    # Include line intersections at the Recall boundary on the logarithmic axis.
    for points in drawn_series.values():
        for a, b in zip(points, points[1:]):
            for boundary in (xmin, xmax):
                if min(a[1], b[1]) < boundary < max(a[1], b[1]):
                    fraction = (boundary - a[1]) / (b[1] - a[1])
                    visible_rates.append(math.exp(math.log(a[2]) * (1 - fraction) + math.log(b[2]) * fraction))
    if not visible_rates:
        raise ValueError(f"No measurements in the requested Recall range: {dataset} {kind}")
    ymin, ymax = min(visible_rates), max(visible_rates)
    # Show at least two explicit powers of ten, including the bottom decade.
    bottom_decade = 10 ** math.floor(math.log10(ymin))
    lower_limit = bottom_decade * .95
    upper_limit = max(ymax * 1.35, bottom_decade * 10 * 1.05)
    for axis, batch in zip(axes, BATCHES):
        for method in METHODS:
            points = drawn_series[method, batch]
            axis.plot([p[1] for p in points], [p[2] for p in points], label=labels[method],
                      linewidth=2, markersize=4.6, markeredgewidth=1.05, **styles[method])
        axis.text(.965, .955, f"QBatch = {batch}", transform=axis.transAxes,
                  ha="right", va="top", fontsize=12,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))
        axis.text(.035, .04, label, transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=12,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))
        axis.set_xlabel("Recall@10", labelpad=8)
        axis.set_xlim(xmin, xmax)
        axis.set_yscale("log")
        axis.set_ylim(lower_limit, upper_limit)
        axis.xaxis.set_major_locator(MultipleLocator(.02) if flat else
                                         MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=12))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(True, linestyle="--", linewidth=.65, alpha=.28)
        axis.grid(True, which="minor", axis="y", linestyle=":", linewidth=.45, alpha=.16)
    axes[0].set_ylabel("QPS", labelpad=10)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .99),
                  ncol=2, frameon=False, fontsize=11, handlelength=2.6, columnspacing=3)
    stem = f"{label}_{'IVFFlat' if flat else 'IVFPQ'}_QBatch128_256_512_K10"
    for extension, options in (("png", {"dpi": 220}), ("pdf", {})):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, **options)
        print(path, flush=True)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot/Figures")
    parser.add_argument("--datasets", nargs="+", default=["glove1.2m", "sift1m"])
    parser.add_argument("--curve-mode", choices=("frontier", "all"), default="frontier")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for kind in ("ivfflat", "ivfpq"):
            make_figure(dataset, kind, args.results_dir, args.output_dir, args.curve_mode)


if __name__ == "__main__":
    main()
