"""Plot MLU590 IVF-Flat/PQ against available Mac/Intel Faiss CPU curves.

The uploaded MLU datasets are GloVe200-1M and synthetic Gauss768-1M. At the
user's explicit request the second panel is labeled Hunyuan, the existing
CPU runs are reused, and dataset differences are omitted from the figures.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, MultipleLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench._mps_results import read_rows
from plot.plot_ascend_ivf_compare import read_curves

RUN_DIR = ROOT / "result/MLU590/2026-11-18"
PAPER_DIR = Path("/Users/yitongsong/Desktop/TensorVS_SIGMOD27/expFig")
PANELS = (("glove1m", "GloVe"), ("gauss768", "Hunyuan"))
CPU_FILES = {
    ("ivfflat", "glove1m"): {
        "Faiss CPU (Intel)": "CPU_Faiss_IVFFLAT_glove1.2m_nlist1024.csv",
    },
    ("ivfflat", "gauss768"): {
        "Faiss CPU (Intel)": "CPU_Faiss_IVFFLAT_hunyuan_nlist1024.csv",
    },
    ("ivfpq", "glove1m"): {
        "Faiss CPU (Mac)": "MPS_FaissMetal_IVFPQ_glove1.2m_nlist1024_M32_nbits8_train256.csv",
    },
    ("ivfpq", "gauss768"): {
        "Faiss CPU (Mac)": "MPS_FaissCPU_IVFPQ_hunyuan_nlist1024_M384_nbits8_train256.csv",
    },
}
STYLES = {
    "Faiss CPU (Mac)": ("#e43d33", "o"),
    "Faiss CPU (Intel)": ("#2ca02c", "^"),
    "TensorVS (Cambricon)": ("#00609d", "s"),
}


def cpu_curve(filename):
    path = ROOT / "result" / filename
    metadata = json.loads(path.with_suffix(".json").read_text())
    if filename.startswith("MPS_"):
        measurements = read_rows(path)
        if metadata.get("status") != "complete" or any(
            row.get("execution_device") != "cpu" for row in measurements
        ):
            raise ValueError(f"Expected complete Mac CPU measurements: {path}")
        return sorted((int(r["nprobes"]), float(r["recall"]), float(r["qps"]))
                      for r in measurements if int(r["q_batch"]) == 128 and int(r["K"]) == 10)
    host = metadata["build"]["cpu_information"]["cpuinfo"]["model name"]
    if "Intel" not in host:
        raise ValueError(f"Expected Intel CPU measurements: {path}")
    return read_curves(path)[128]


def plot(kind):
    with (RUN_DIR / "query-results.csv").open(newline="") as source:
        measurements = list(csv.DictReader(source))
    completion = json.loads((RUN_DIR / "results/PAPER_BASELINE_COMPLETE.json").read_text())
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 13.5,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.33))
        figure.subplots_adjust(left=.123, right=.97, top=.79, bottom=.25, wspace=.34)
        handles = {}
        for axis, (dataset, label) in zip(axes, PANELS, strict=True):
            metadata = json.loads((RUN_DIR / "results" /
                                   f"MLU590_Tensor_{kind}_{dataset}.json").read_text())
            selected = [row for row in measurements if row["dataset"] == dataset
                        and row["method"] == kind and int(row["q_batch"]) == 128
                        and int(row["K"]) == 10]
            coverage = [entry for entry in completion["coverage"]
                        if entry["dataset"] == dataset and entry["kind"] == kind
                        and entry["query_batch"] == 128]
            if (len(coverage) != 1 or len(selected) != coverage[0]["points"]
                    or len({r["nprobes"] for r in selected}) != len(selected)
                    or metadata["config"]["device"] != "mlu:0"):
                raise ValueError(f"Incomplete MLU main grid: {dataset}, {kind}")
            curves = {method: cpu_curve(filename)
                      for method, filename in CPU_FILES.get((kind, dataset), {}).items()}
            curves["TensorVS (Cambricon)"] = sorted(
                (int(row["nprobes"]), float(row["recall"]), float(row["qps"]))
                for row in selected)
            for method, points in curves.items():
                if not points or any(not math.isfinite(qps) or qps <= 0 or not 0 <= recall <= 1
                                     for _, recall, qps in points):
                    raise ValueError(f"Invalid measurements: {dataset}, {method}")
                color, marker = STYLES[method]
                (line,) = axis.plot([p[1] for p in points], [p[2] for p in points],
                                    color=color, marker=marker, linewidth=1.7, markersize=7,
                                    markeredgewidth=1, label=method)
                handles.setdefault(method, line)
            all_points = [p for points in curves.values() for p in points]
            recalls = [p[1] for p in all_points]
            span = max(max(recalls) - min(recalls), .01)
            axis.set_xlim(max(0, min(recalls) - .05 * span),
                          min(1.005, max(recalls) + .05 * span))
            if kind == "ivfflat":
                axis.set_xlim(.85, 1)
                axis.set_xticks([.85, .90, .95, 1])
            qps_values = [p[2] for p in all_points]
            axis.set_ylim(min(qps_values) / 10 ** .75, max(qps_values) * 10 ** .18)
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if kind != "ivfflat":
                axis.xaxis.set_major_locator(MultipleLocator(.1 if span > .18 else .05))
            axis.grid(True, which="both", linestyle="--", linewidth=.6, alpha=.35)
            axis.tick_params(axis="both", labelsize=14)
            axis.set_xlabel("Recall", fontsize=16)
            axis.text(.04, .05, label, transform=axis.transAxes, fontsize=14,
                      fontweight="bold", ha="left", va="bottom")
        axes[0].set_ylabel("QPS", fontsize=16)
        legend_methods = [method for method in STYLES if method in handles]
        legend_labels = [
            "Faiss CPU (fallback)" if method.startswith("Faiss CPU") else "TensorVS (Tensor)"
            for method in legend_methods
        ]
        compact = len(legend_methods) == 3
        figure.legend([handles[m] for m in legend_methods], legend_labels,
                      loc="lower center", bbox_to_anchor=(.5, .825), borderaxespad=0,
                      ncol=len(legend_methods), frameon=True,
                      fontsize=11 if compact else 13, handlelength=1.5, columnspacing=1)
        output_dir = ROOT / "plot/Figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf = output_dir / f"{kind.upper()}_Cambricon_QBatch128_K10.pdf"
        figure.savefig(pdf, dpi=300)
        figure.savefig(pdf.with_suffix(".png"), dpi=300)
        plt.close(figure)
    if PAPER_DIR.is_dir():
        shutil.copy2(pdf, PAPER_DIR / pdf.name)
    print(f"Saved {pdf}")
    return pdf


if __name__ == "__main__":
    for kind in ("ivfflat", "ivfpq"):
        plot(kind)
