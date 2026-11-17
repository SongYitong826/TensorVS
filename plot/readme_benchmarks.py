"""Redraw the README's GloVe and Hunyuan figures with the paper's plotting styles."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from plot import mps_kernel_performance as apple
from plot import plot_amd_ivf_compare as amd
from plot import plot_ascend_ivf_compare as ascend
from plot.ivfflat_cuda_kernel import LEGEND_ORDER, METHODS

ROOT = Path(__file__).resolve().parents[1]
NVIDIA_CURVES = ROOT / "docs/benchmark-data/nvidia-paper-curves.csv"
DATASETS = (("glove1.2m", "GloVe"), ("hunyuan", "Hunyuan"))
KINDS = ("ivfflat", "ivfpq")
BATCHES = (32, 128, 512)
NVIDIA_AXES = {
    ("ivfflat", "glove1.2m"): ((.90, 1.00), (.90, .92, .94, .96, .98, 1.), (926, 29990)),
    ("ivfflat", "hunyuan"): ((.90, 1.00), (.90, .92, .94, .96, .98, 1.), (100, 10000)),
    ("ivfpq", "glove1.2m"): ((.34, .38), (.34, .35, .36, .37, .38), (11510, 503500)),
    ("ivfpq", "hunyuan"): ((.46, .79), (.50, .60, .70), (15970, 102000)),
}


def load_nvidia_curves(source: Path):
    """Read points recovered from the plotted curves of the two CUDA paper figures."""
    curves = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            key = row["index"], row["dataset"], int(row["q_batch"]), row["method"]
            curves[key].append((float(row["recall"]), float(row["qps"])))
    return curves


def plot_nvidia(curves, kind: str, output: Path) -> None:
    """Draw only the paper's GloVe and Hunyuan panels with the original axes."""
    flat = kind == "ivfflat"
    batches = BATCHES if flat else (128,)
    with plt.rc_context({"font.family": "DejaVu Sans", "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(len(batches), 2, figsize=(7.4, 6.8 if flat else 2.75),
                                   squeeze=False)
        figure.subplots_adjust(left=.105, right=.97, top=.895 if flat else .77,
                               bottom=.095 if flat else .23, wspace=.33, hspace=.12)
        handles = {}
        for row, batch in enumerate(batches):
            for col, (dataset, label) in enumerate(DATASETS):
                axis = axes[row, col]
                for method_key, method in METHODS.items():
                    if kind == "ivfpq" and dataset == "hunyuan" and method_key == "faiss":
                        continue
                    points = curves.get((kind, dataset, batch, method_key))
                    if not points:
                        raise ValueError(f"Missing {kind}, {dataset}, QBatch={batch}, {method.label}")
                    recall, qps = zip(*points)
                    line, = axis.plot(recall, qps, color=method.color, marker=method.marker,
                                      linestyle=method.linestyle, lw=1.5, ms=6.4,
                                      label=method.label)
                    handles[method_key] = line

                recall_limits, recall_ticks, qps_limits = NVIDIA_AXES[kind, dataset]
                axis.set_xlim(*recall_limits)
                axis.set_xticks(recall_ticks)
                axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                axis.set_yscale("log")
                axis.set_ylim(*qps_limits)
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                axis.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
                axis.yaxis.set_minor_formatter(NullFormatter())
                axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
                axis.tick_params(labelsize=11)
                bottom_row = row == len(batches) - 1
                axis.tick_params(axis="x", labelbottom=bottom_row)
                if bottom_row:
                    axis.set_xlabel("Recall", fontsize=13)
                if col == 0:
                    axis.set_ylabel("QPS", fontsize=13)

                annotation = {"transform": axis.transAxes, "fontsize": 12,
                              "fontweight": "bold",
                              "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": .78}}
                axis.text(.04, .06, label, va="bottom", **annotation)
                if flat:
                    axis.text(.96, .94, f"QBatch = {batch}", ha="right", va="top", **annotation)

        figure.legend([handles[key] for key in LEGEND_ORDER],
                      [METHODS[key].label for key in LEGEND_ORDER], loc="upper center",
                      bbox_to_anchor=(.5, .99), ncols=len(LEGEND_ORDER),
                      fontsize=13, handlelength=2, columnspacing=1.2)
        figure.savefig(output, dpi=220)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--nvidia-curves", type=Path, default=NVIDIA_CURVES)
    args = parser.parse_args()
    curves = load_nvidia_curves(args.nvidia_curves)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temporary:
        work = Path(temporary)
        for kind in KINDS:
            plot_nvidia(curves, kind, work / f"benchmark-nvidia-{kind}.png")
            # Match the batch sizes selected for each figure in the paper.
            batch = None if kind == "ivfflat" else 128
            _, amd_png = amd.plot(args.result_dir, work, kind, batch, sync_paper=False)
            _, apple_png = apple.plot_kind(kind, args.result_dir, work, batch, sync_paper=False)
            _, ascend_png = ascend.plot(args.result_dir, work, kind, 128, sync_paper=False)
            for platform, png in (("amd", amd_png), ("apple", apple_png), ("ascend", ascend_png)):
                png.rename(work / f"benchmark-{platform}-{kind}.png")

        for png in sorted(work.glob("benchmark-*.png")):
            target = args.output_dir / png.name
            shutil.copy2(png, target)
            print(f"Saved {target}")


if __name__ == "__main__":
    main()
