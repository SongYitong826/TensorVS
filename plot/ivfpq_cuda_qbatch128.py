"""Redraw the CUDA IVF-PQ QBatch=128 curves from the vector paper figure.

Some baseline CSVs are only on the CUDA server. The source PDF stores the
plotted points as vector paths. Convert those coordinates back to Recall and
QPS using the source axes, then render the six panels and text at native size.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from pypdf import PdfReader

if __package__:
    from .ivfpq_cuda_kernel import LEGEND_ORDER, METHODS
else:
    from ivfpq_cuda_kernel import LEGEND_ORDER, METHODS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "plot/Figures/IVFPQ_CUDA_QBatch32_128_512_K10.pdf"
PAPER_DIR = Path("/Users/yitongsong/Desktop/TensorVS_SIGMOD27/expFig")
SOURCE_PANEL_SIZE = (166.2759, 129.7582)


@dataclass(frozen=True)
class Panel:
    label: str
    origin: tuple[float, float]
    recall_ticks: tuple[float, ...]
    recall_tick_x: tuple[float, float]
    qps_ticks: tuple[float, float]
    qps_tick_y: tuple[float, float]
    point_count: int


# Two source tick positions calibrate each axis. The QPS axis is logarithmic.
# SIFT was replaced in the source PDF and has a different panel origin.
PANELS = (
    Panel("SIFT", (61.156, 190.6348), (.37, .38, .39, .40, .41),
          (63.12927, 100.36077), (1e3, 1e6), (190.6348, 320.393), 14),
    Panel("GloVe", (265.3704, 215.9848), (.34, .35, .36, .37, .38),
          (265.37042, 306.93940), (1e5, 2e5), (290.25597, 314.04104), 18),
    Panel("Tiny", (476.5408, 215.9848), (.50, .52, .54, .56, .58, .60),
          (476.54083, 509.79602), (1e4, 1e5), (243.69007, 303.08205), 19),
    Panel("Hunyuan", (687.7112, 215.9848), (.50, .60, .70),
          (707.86591, 758.25255), (2e4, 3e4), (232.01203, 260.94324), 15),
    Panel("GIST", (898.8817, 215.9848), (.45, .50, .55),
          (946.38907, 1005.77333), (1e4, 1e5), (223.53161, 292.56368), 15),
    Panel("OpenAI", (1110.0521, 215.9848), (.60, .65, .70, .75),
          (1110.05208, 1156.23984), (1e4, 1e5), (236.08305, 306.12294), 15),
)
METHOD_BY_RGB = {
    tuple(round(int(method.color[i:i + 2], 16) / 255, 3) for i in (1, 3, 5)): key
    for key, method in METHODS.items()
}


def read_paths(source: Path):
    reader = PdfReader(source)
    if len(reader.pages) != 1 or tuple(map(float, reader.pages[0].mediabox)) != (0, 0, 1297.44, 520):
        raise ValueError("Expected the fixed-layout CUDA IVF-PQ paper figure")
    operations = reader.pages[0].get_contents().operations
    panel_by_origin = {panel.origin: panel for panel in PANELS}
    curves = {panel.label: {} for panel in PANELS}
    active_panel = None
    stack = []

    for i, (operands, operator) in enumerate(operations):
        if operator == b"q":
            stack.append(active_panel)
        elif operator == b"Q":
            active_panel = stack.pop()
        elif operator == b"re" and operations[i + 1][1] == b"W":
            x, y, width, height = (round(float(value), 4) for value in operands)
            active_panel = (panel_by_origin.get((x, y))
                            if (width, height) == SOURCE_PANEL_SIZE else None)
        elif operator == b"RG" and active_panel is not None:
            method = METHOD_BY_RGB.get(tuple(round(float(value), 3) for value in operands))
            if method is None:
                continue
            start = i + 1
            while operations[start][1] in (b"cs", b"rg", b"gs", b"j", b"w"):
                start += 1
            if operations[start][1] != b"m":
                continue
            end = start + 1
            while operations[end][1] == b"l":
                end += 1
            if operations[end][1] != b"S" or end - start != active_panel.point_count:
                continue
            # Later overlays replace the SIFT panel with the final paper results.
            curves[active_panel.label][method] = [
                tuple(map(float, args)) for args, _ in operations[start:end]
            ]

    for panel in PANELS:
        expected = set(METHODS) - ({"faiss"} if panel.label == "Hunyuan" else set())
        if curves[panel.label].keys() != expected:
            raise ValueError(f"Unexpected vector curves in source panel {panel.label}")
    return curves


def plot(source: Path, output_dir: Path) -> Path:
    curves = read_paths(source)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        figure, axes = plt.subplots(2, 3, figsize=(9.8, 4.34))
        figure.subplots_adjust(left=.085, right=.975, top=.79, bottom=.145,
                               wspace=.30, hspace=.28)
        handles = {}
        for index, (panel, axis) in enumerate(zip(PANELS, axes.flat)):
            left, bottom = panel.origin
            width, height = SOURCE_PANEL_SIZE
            x0, x1 = panel.recall_tick_x
            recall0, recall1 = panel.recall_ticks[:2]
            recall_per_point = (recall1 - recall0) / (x1 - x0)
            y0, y1 = panel.qps_tick_y
            log_qps0, log_qps1 = map(math.log10, panel.qps_ticks)
            log_qps_per_point = (log_qps1 - log_qps0) / (y1 - y0)
            qps_low = 10 ** (log_qps0 + (bottom - y0) * log_qps_per_point)
            qps_high = 10 ** (log_qps0 + (bottom + height - y0) * log_qps_per_point)

            for key, points in curves[panel.label].items():
                method = METHODS[key]
                (line,) = axis.plot(
                    [recall0 + (x - x0) * recall_per_point for x, _ in points],
                    [10 ** (log_qps0 + (y - y0) * log_qps_per_point) for _, y in points],
                    color=method.color, marker=method.marker, linestyle=method.linestyle,
                    linewidth=1.7, markersize=6.4,
                )
                handles.setdefault(key, line)

            axis.set_xticks(panel.recall_ticks)
            axis.set_xlim(min(panel.recall_ticks[0], recall0 + (left - x0) * recall_per_point),
                          max(panel.recall_ticks[-1], recall0 + (left + width - x0) * recall_per_point))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.set_yscale("log")
            if panel.label == "Hunyuan":
                qps_high = max(qps_high, 1e5 * 1.02)
            axis.set_ylim(qps_low, qps_high)
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
            axis.tick_params(axis="both", labelsize=13)
            if index % 3 == 0:
                axis.set_ylabel("QPS", fontsize=16)
            if index >= 3:
                axis.set_xlabel("Recall", fontsize=16)
            label_x = .32 if panel.label == "OpenAI" else .04
            axis.text(label_x, .05, panel.label, transform=axis.transAxes, fontsize=15,
                      fontweight="bold", va="bottom",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .78})

        figure.legend([handles[key] for key in LEGEND_ORDER],
                      ["cuVS", "Faiss GPU", "TensorVS (Triton)"],
                      loc="lower center", bbox_to_anchor=(.5, .833), borderaxespad=0,
                      ncol=3, fontsize=16, handlelength=2, columnspacing=1.2)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf = output_dir / "IVFPQ_CUDA_QBatch128_K10.pdf"
        figure.savefig(pdf, dpi=300)
        figure.savefig(pdf.with_suffix(".png"), dpi=300)
        plt.close(figure)
    if PAPER_DIR.is_dir() and (PAPER_DIR / pdf.name).resolve() != pdf.resolve():
        shutil.copy2(pdf, PAPER_DIR / pdf.name)
    print(f"Saved {pdf}")
    return pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot/Figures")
    args = parser.parse_args()
    plot(args.source, args.output_dir)
