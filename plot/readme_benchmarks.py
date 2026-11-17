"""Render GloVe and Hunyuan benchmark cards for the TensorVS README."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, LogLocator, MaxNLocator, NullFormatter, PercentFormatter
from PIL import Image

from plot import plot_amd_ivf_compare as amd
from plot import plot_ascend_ivf_compare as ascend
from plot.mps_kernel_performance import performance_frontier, read_curves as read_mps_curves

ROOT = Path(__file__).resolve().parents[1]
BATCH = 128
KINDS = (("ivfflat", "IVF-Flat"), ("ivfpq", "IVF-PQ"))
TENSOR = "TensorVS (Tensor)"
TRITON = "TensorVS (Triton)"
MPS_CPU = "Faiss Metal package (CPU)"
CPU = "Faiss CPU"
MAC_CPU = "Faiss CPU (Mac)"
STYLES = {
    TENSOR: ("#1760b5", "s"),
    TRITON: ("#087f76", "D"),
    "hipVS": ("#b86611", "^"),
    "Faiss ROCm": ("#a43d66", "o"),
    "Faiss Metal": ("#a43d66", "o"),
    MPS_CPU: ("#a43d66", "P"),
    CPU: ("#786198", "^"),
    MAC_CPU: ("#786198", "v"),
}


def amd_curves(result_dir, dataset, kind, pq_config):
    return {
        method: amd.read_curves(amd.result_path(result_dir, method, kind, dataset, pq_config))[BATCH]
        for method in amd.methods_for(dataset, kind)
    }


def mps_curves(result_dir, dataset, kind, _pq_config):
    curves, rows = read_mps_curves(result_dir, dataset, kind)
    labels = {"TensorVS": TENSOR, "Faiss Metal": "Faiss Metal",
              "FAISS CPU (fallback)": CPU}
    devices = {row["execution_device"] for row in rows
               if row["method"] == "Faiss Metal" and int(row["q_batch"]) == BATCH}
    if devices == {"cpu"}:
        labels["Faiss Metal"] = MPS_CPU
    return {labels[method]: points for (method, batch), points in curves.items() if batch == BATCH}


def ascend_curves(result_dir, dataset, kind, pq_config):
    tensor = ascend.result_path(result_dir, ascend.TENSOR_METHOD, kind, dataset, pq_config)
    curves = {TENSOR: ascend.read_curves(tensor)[BATCH]}
    # The GloVe Faiss Ascend PQ run uses IP ground truth; the Tensor run uses L2.
    if kind == "ivfpq" and dataset == "glove1.2m":
        return curves
    baseline = ascend.result_path(result_dir, ascend.FAISS_METHOD, kind, dataset, pq_config)
    curves[MAC_CPU if kind == "ivfpq" else CPU] = ascend.read_curves(baseline)[BATCH]
    return curves


def render_card(destination, result_dir, load_curves):
    with plt.rc_context({
        "font.family": "DejaVu Sans", "text.color": "#233247",
        "axes.labelcolor": "#233247", "axes.edgecolor": "#c6ceda",
        "xtick.color": "#42516a", "ytick.color": "#42516a",
    }):
        figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.15))
        figure.subplots_adjust(left=.10, right=.98, bottom=.115, top=.855,
                               wspace=.24, hspace=.62)
        handles = {}
        for row, (kind, kind_label) in enumerate(KINDS):
            for column, (dataset, dataset_label, pq_config) in enumerate(amd.DATASETS):
                axis = axes[row, column]
                methods = load_curves(result_dir, dataset, kind, pq_config)
                for method, measurements in methods.items():
                    visible = [point for point in measurements if kind != "ivfflat" or point[1] >= .85]
                    if not visible:
                        raise ValueError(f"No visible points for {method}, {dataset}, {kind}")
                    points = sorted(visible) if kind == "ivfflat" else performance_frontier(visible)
                    _, recall, qps = zip(*points)
                    color, marker = STYLES[method]
                    line, = axis.plot(recall, qps, color=color, marker=marker,
                                      lw=1.8, ms=4.1, label=method)
                    handles[method] = line
                axis.set_title(f"{dataset_label}  /  {kind_label}", loc="left",
                               fontsize=11, fontweight="bold", pad=7)
                axis.set_yscale("log")
                axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
                axis.yaxis.set_major_formatter(EngFormatter(sep=""))
                axis.yaxis.set_minor_formatter(NullFormatter())
                axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
                axis.xaxis.set_major_locator(MaxNLocator(5))
                axis.grid(axis="y", color="#e4e9ef", linewidth=.7)
                axis.set_axisbelow(True)
                if kind == "ivfflat":
                    axis.set_xlim(.845, 1.006)
                if column == 0:
                    axis.set_ylabel("Queries / second")
                axis.set_xlabel("Recall@10")
        legend = [handles[method] for method in STYLES if method in handles]
        figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .985),
                      ncols=len(legend), frameon=False, fontsize=9)
        figure.savefig(destination, dpi=170, facecolor="white")
        plt.close(figure)


def render_nvidia(destination, flat_pdf, pq_pdf):
    """Keep the drawn NVIDIA curves intact, using the axes of the paper panels."""
    with TemporaryDirectory() as temporary:
        sources = {}
        for kind, source in (("ivfflat", flat_pdf), ("ivfpq", pq_pdf)):
            prefix = Path(temporary) / kind
            subprocess.run(["pdftoppm", "-r", "216", "-png", "-singlefile", str(source), str(prefix)],
                           check=True)
            with Image.open(prefix.with_suffix(".png")) as page:
                sources[kind] = page.convert("RGB")

        def crop(kind, bounds):
            # Bounds use PDF points from the top left; render scale is 3 px/pt.
            return sources[kind].crop(tuple(round(value * 3) for value in bounds))

        legend = crop("ivfflat", (387, 5, 909, 38))
        flat_glove = crop("ivfflat", (230, 188, 447, 338))
        flat_hunyuan = crop("ivfflat", (652, 188, 869, 338))
        pq_glove = crop("ivfpq", (256, 61, 482, 174))
        pq_hunyuan = crop("ivfpq", (25, 174, 251, 287))

        figure = plt.figure(figsize=(8.6, 5.15), facecolor="white")
        for bounds, panel, recall_axis in (
            ((.205, .912, .59, .062), legend, None),
            ((.071, .550, .43, .298), flat_glove, 265.3704 - 230),
            ((.543, .550, .43, .298), flat_hunyuan, 687.7112 - 652),
            ((.071, .079, .43, .325), pq_glove, None),
            ((.543, .079, .43, .325), pq_hunyuan, None),
        ):
            axis = figure.add_axes(bounds)
            axis.imshow(panel, aspect="auto", interpolation="lanczos")
            axis.set_axis_off()
            if recall_axis is not None:
                # The paper labels the shared .90–1.00 ticks on the bottom row.
                for tick in range(6):
                    x = (recall_axis + tick * 166.2759 / 5) / 217
                    axis.text(x, -.01, f"{.90 + .02 * tick:.2f}",
                              transform=axis.transAxes, ha="center", va="top", fontsize=9)
        for y, title in ((.861, "IVF-Flat"), (.429, "IVF-PQ")):
            figure.text(.50, y, title, ha="center", fontsize=12, fontweight="bold",
                        color="#233247")
        for y in (.694, .264):
            figure.text(.033, y, "Queries / second", ha="center", va="center",
                        rotation=90, fontsize=10, color="#233247")
        for y in (.493, .046):
            figure.text(.50, y, "Recall@10", ha="center", fontsize=10, color="#233247")
        figure.savefig(destination, dpi=170)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--nvidia-flat", type=Path,
                        default=ROOT / "plot/Figures/IVFFlat_CUDA_QBatch32_128_512_K10.pdf")
    parser.add_argument("--nvidia-pq", type=Path,
                        default=ROOT / "plot/Figures/IVFPQ_CUDA_QBatch128_K10.pdf")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for platform, loader in (("amd", amd_curves), ("apple", mps_curves), ("ascend", ascend_curves)):
        render_card(args.output_dir / f"benchmark-{platform}.png", args.result_dir, loader)
    render_nvidia(args.output_dir / "benchmark-nvidia.png", args.nvidia_flat, args.nvidia_pq)


if __name__ == "__main__":
    main()
