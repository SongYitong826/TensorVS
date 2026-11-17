"""Plot IVFFlat QPS sensitivity to K at a fixed Recall target.

The default paper configuration is one row containing GloVe, Hunyuan, and
OpenAI at QBatch=128 and Recall=0.99.  QPS is linearly interpolated between
the two measured nprobe points bracketing the Recall target.

Run from the TensorVS repository root with::

    conda run -n tensorvs python plot/ivfflat_k_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from ivfflat_cuda_kernel import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULT_DIRS,
    METHODS,
    ResultFile,
    discover_result_files,
    read_result_rows,
)


DATASETS = (
    ("glove1.2m", "GloVe"),
    ("hunyuan", "Hunyuan"),
    ("openai1536", "OpenAI"),
)
K_VALUES = (20, 40, 60, 80, 100)
LEGEND_ORDER = ("cuvs", "faiss", "tensorvs")
DEFAULT_QBATCH = 128
DEFAULT_RECALL = 0.99
PAPER_POINTS = Path(__file__).with_name("ivfflat_k_sensitivity_paper_points.json")
PAPER_FIGURE_DIR = Path.home() / "Desktop" / "TensorVS_SIGMOD27" / "expFig"


def interpolate_qps_at_recall(
    rows: list[dict[str, float]], target_recall: float
) -> float | None:
    """Return linearly interpolated QPS at the requested Recall."""

    qps_by_recall: dict[float, float] = {}
    for row in rows:
        recall = row["recall"]
        qps_by_recall[recall] = max(qps_by_recall.get(recall, 0.0), row["qps"])
    points = sorted(qps_by_recall.items())
    if not points or target_recall < points[0][0] or target_recall > points[-1][0]:
        return None

    for recall, qps in points:
        if math.isclose(recall, target_recall, rel_tol=0.0, abs_tol=1e-12):
            return qps
    lower = next(point for point in reversed(points) if point[0] < target_recall)
    upper = next(point for point in points if point[0] > target_recall)
    ratio = (target_recall - lower[0]) / (upper[0] - lower[0])
    return lower[1] + ratio * (upper[1] - lower[1])


def collect_points(
    result_files: dict[str, dict[str, ResultFile]],
    q_batch: int,
    target_recall: float,
) -> tuple[dict[str, dict[str, dict[int, float]]], list[str]]:
    """Collect dataset -> method -> K -> interpolated QPS."""

    points = {dataset: {} for dataset, _label in DATASETS}
    warnings: list[str] = []
    for method in METHODS:
        for dataset, label in DATASETS:
            result_file = result_files[method].get(dataset)
            if result_file is None:
                warnings.append(f"{METHODS[method].label}/{label}: file missing")
                continue
            try:
                rows = read_result_rows(result_file)
            except (OSError, ValueError) as error:
                warnings.append(str(error))
                continue

            method_points: dict[int, float] = {}
            for topk in K_VALUES:
                matching = [
                    row for row in rows
                    if int(row["q_batch"]) == q_batch and int(row["topk"]) == topk
                ]
                qps = interpolate_qps_at_recall(matching, target_recall)
                if qps is None:
                    warnings.append(
                        f"{METHODS[method].label}/{label}: no Recall={target_recall:.2f} "
                        f"bracket for QBatch={q_batch}, K={topk}"
                    )
                else:
                    method_points[topk] = qps
            if method_points:
                points[dataset][method] = method_points
    return points, warnings


@plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
def plot_figure(
    points: dict[str, dict[str, dict[int, float]]],
    q_batch: int,
    target_recall: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, len(DATASETS), figsize=(840 / 72, 230 / 72), squeeze=False)
    axes = axes[0]
    legend_handles: dict[str, object] = {}

    for column, (dataset, label) in enumerate(DATASETS):
        axis = axes[column]
        dataset_points = points[dataset]
        for method_key, method in METHODS.items():
            method_points = dataset_points.get(method_key, {})
            k_values = [topk for topk in K_VALUES if topk in method_points]
            if not k_values:
                continue
            (line,) = axis.plot(
                k_values,
                [method_points[topk] for topk in k_values],
                color=method.color,
                marker=method.marker,
                linestyle=method.linestyle,
                linewidth=1.5,
                markersize=6.4,
                markerfacecolor=method.markerfacecolor,
                markeredgecolor=method.color,
                markeredgewidth=1.0,
                label=method.label,
            )
            legend_handles.setdefault(method_key, line)

        axis.set_xticks(K_VALUES)
        axis.set_xlim(K_VALUES[0] - 5, K_VALUES[-1] + 5)
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12)
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("k", fontsize=23)
        axis.set_ylabel("QPS" if column == 0 else "", fontsize=23)
        axis.tick_params(axis="both", labelsize=18)
        axis.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.text(
            0.04,
            0.49 if dataset == "glove1.2m" else 0.42,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=21,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        if not dataset_points:
            axis.set_ylim(1e2, 1e4)
            axis.text(
                0.5,
                0.5,
                "Results pending",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#777777",
            )

    legend_keys = [key for key in LEGEND_ORDER if key in legend_handles]
    if legend_keys:
        figure.legend(
            [legend_handles[key] for key in legend_keys],
            [METHODS[key].label for key in legend_keys],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(legend_keys),
            frameon=True,
            fontsize=20,
            handlelength=2.2,
            columnspacing=1.5,
        )
    figure.subplots_adjust(
        left=0.09, right=0.96, bottom=0.227, top=0.72, wspace=0.33
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    recall_tag = f"{target_recall:.2f}".replace(".", "p")
    stem = f"IVFFlat_SensitiveToK_QBatch{q_batch}_Recall{recall_tag}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, dpi=300)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-batch", type=int, default=DEFAULT_QBATCH)
    parser.add_argument("--recall", type=float, default=DEFAULT_RECALL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.recall <= 1.0:
        raise ValueError("--recall must be in (0, 1]")
    points, warnings = collect_points(
        discover_result_files(DEFAULT_RESULT_DIRS), args.q_batch, args.recall
    )
    if warnings:
        print("Incomplete result entries:")
        for warning in warnings:
            print(f"  - {warning}")
        published = json.loads(PAPER_POINTS.read_text())
        if (args.q_batch, args.recall) == (published["q_batch"], published["target_recall"]):
            points = {
                dataset: {
                    method: {int(k): qps for k, qps in series.items()}
                    for method, series in methods.items()
                }
                for dataset, methods in published["points"].items()
            }
            print(f"Using the complete set of curves recovered from {published['source']}")
    pdf_path, png_path = plot_figure(
        points, args.q_batch, args.recall, args.output_dir
    )
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")
    if (args.q_batch, args.recall) == (DEFAULT_QBATCH, DEFAULT_RECALL) and PAPER_FIGURE_DIR.is_dir():
        paper_pdf = PAPER_FIGURE_DIR / pdf_path.name
        if paper_pdf.resolve() != pdf_path.resolve():
            shutil.copy2(pdf_path, paper_pdf)
            print(f"Synced {paper_pdf}")


if __name__ == "__main__":
    main()
