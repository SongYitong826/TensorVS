"""Replace SIFT panels in the existing fixed-layout CUDA paper PDFs."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
TOKENS = {"tensorvs": "TensorVS_Triton", "cuvs": "cuVS", "faiss": "FaissGPU"}
STYLES = {"tensorvs": ("#00609d", "s"), "cuvs": ("#f5ba03", "o"), "faiss": ("#e43d33", "^")}
IVFPQ_QPS_LIMITS = (1e3, 1e6)


def sift_file(directory, method, kind):
    paths = sorted(directory.glob(f"CUDA_{TOKENS[method]}_MultiGPU3_{kind.upper()}_sift100m_nlist*.csv"))
    if len(paths) > 1:
        raise ValueError(f"Ambiguous SIFT100M results: {paths}")
    if not paths:
        return None
    metadata = json.loads(paths[0].with_suffix(".json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError(f"Incomplete SIFT100M experiment: {paths[0]}")
    return paths[0]


def update_sift_column(kind, output_dir, output_stem, q_batches, topk, recall_min=.94, recall_max=1., result_dir=None):
    """Keep the existing paper composition and replace its three SIFT axes."""
    if tuple(q_batches) != (32, 128, 512) or topk != 10:
        raise ValueError("The existing paper composition requires QBatch=32/128/512 and K=10")
    pdf = output_dir / f"{output_stem}.pdf"
    original = PdfReader(pdf)
    page = original.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    if len(original.pages) != 1 or abs(width - 1297.44) > .1 or abs(height - 520) > .1:
        raise ValueError("Unexpected paper canvas; regenerate the full figure for this layout")
    curves = {batch: {} for batch in q_batches}
    source_dir = Path(result_dir) if result_dir is not None else ROOT / "result"
    for method in TOKENS:
        path = sift_file(source_dir, method, kind)
        if path is None:
            raise FileNotFoundError(f"Missing SIFT100M {method}/{kind}")
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        for batch in q_batches:
            selected = [r for r in rows if int(r["q_batch"]) == batch and int(r["K"]) == topk]
            if len(selected) != 14 or len({r["nprobes"] for r in selected}) != 14:
                raise ValueError(f"Incomplete SIFT100M curve: {path}, QBatch={batch}")
            points = sorted((float(r["recall"]), float(r["qps"])) for r in selected)
            if kind == "ivfflat":
                # Match the full figure's interpolated recall boundary.
                lower = [p for p in points if p[0] < recall_min]
                upper = [p for p in points if p[0] > recall_min]
                kept = [p for p in points if recall_min <= p[0] <= recall_max]
                if lower and upper and not any(p[0] == recall_min for p in kept):
                    lo, hi = lower[-1], upper[0]
                    kept.insert(0, (recall_min, lo[1] + (hi[1] - lo[1]) * (recall_min - lo[0]) / (hi[0] - lo[0])))
                points = kept
            curves[batch][method] = points
    all_points = [p for methods in curves.values() for points in methods.values() for p in points]
    low_x, high_x = min(p[0] for p in all_points), max(p[0] for p in all_points)
    low_y, high_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
    if kind == "ivfflat":
        left, bottoms, mask_width = 54.2, (342.2657, 192.9848, 44.7039), 236.
    else:
        left, bottoms, mask_width = 61.156, (335.9157, 190.6348, 45.3539), 243.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(width / 72, height / 72))
    figure.add_artist(Rectangle((0, 0), mask_width / width, 1, transform=figure.transFigure,
                                facecolor="white", edgecolor="none", zorder=-10))
    for batch, bottom in zip(q_batches, bottoms):
        axis = figure.add_axes((left / width, bottom / height, 166.2759 / width, 129.7582 / height))
        for method, points in curves[batch].items():
            color, marker = STYLES[method]
            axis.plot([p[0] for p in points], [p[1] for p in points], color=color, marker=marker,
                      linewidth=1.5, markersize=6.4, markeredgewidth=1., markerfacecolor=color)
        axis.set_yscale("log")
        axis.set_ylim(*(IVFPQ_QPS_LIMITS if kind == "ivfpq" else (low_y / 1.2, high_y * 1.2)))
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12))
        axis.yaxis.set_minor_formatter(NullFormatter())
        if kind == "ivfflat":
            axis.set_xlim(recall_min, recall_max)
            axis.set_xticks([recall_min + i * (recall_max - recall_min) / 3 for i in range(4)])
        else:
            span = max(high_x - low_x, .01)
            axis.set_xlim(low_x - .05 * span, high_x + .05 * span)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.tick_params(axis="both", labelsize=12.5)
        axis.tick_params(axis="x", labelbottom=batch == q_batches[-1])
        if batch == q_batches[-1]:
            axis.set_xlabel("Recall", fontsize=14)
        axis.set_ylabel("QPS", fontsize=14)
        axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
        box = {"facecolor": "white", "edgecolor": "none", "alpha": .78}
        axis.text(.04, .06, "SIFT", transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=13.5, fontweight="bold", bbox=box)
        axis.text(.96, .94, f"QBatch = {batch}", transform=axis.transAxes, ha="right", va="top",
                  fontsize=12.5, fontweight="bold", bbox=box)
    overlay = io.BytesIO()
    figure.savefig(overlay, format="pdf", transparent=True)
    plt.close(figure)
    page.merge_page(PdfReader(overlay).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    temporary = pdf.with_suffix(".pdf.tmp")
    with temporary.open("wb") as destination:
        writer.write(destination)
    temporary.replace(pdf)
    subprocess.run(["pdftoppm", "-r", "300", "-singlefile", "-png", str(pdf), str(pdf.with_suffix(""))], check=True)
    print(f"Updated SIFT100M panels in {pdf}")
