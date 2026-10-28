"""Plot IVFPQ QPS sensitivity to K at fixed dataset-specific nprobe.

The paper configuration is QBatch=128, K in {20, 40, 60, 80, 100}, with
nprobe=300/700/200 for GloVe/Hunyuan/OpenAI, respectively.

Run from the TensorVS repository root with::

    conda run -n tensorvs python plot/ivfpq_k_sensitivity.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from ivfpq_cuda_kernel import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULT_DIRS,
    METHODS,
    ResultFile,
    discover_result_files,
    read_result_rows,
)


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    nprobe: int


DATASETS = (
    Dataset("glove1.2m", "GloVe", 300),
    Dataset("hunyuan", "Hunyuan", 700),
    Dataset("openai1536", "OpenAI", 200),
)
K_VALUES = (20, 40, 60, 80, 100)
LEGEND_ORDER = ("cuvs", "faiss", "tensorvs")
DEFAULT_QBATCH = 128


def collect_points(
    result_files: dict[str, dict[str, ResultFile]],
    datasets: tuple[Dataset, ...],
    q_batch: int,
) -> tuple[dict[str, dict[str, dict[int, float]]], list[str]]:
    """Collect dataset -> method -> K -> QPS at the exact fixed nprobe."""

    points = {dataset.key: {} for dataset in datasets}
    warnings: list[str] = []
    for method_key, method in METHODS.items():
        for dataset in datasets:
            result_file = result_files[method_key].get(dataset.key)
            if result_file is None:
                warnings.append(f"{method.label}/{dataset.label}: file missing")
                continue
            try:
                rows = read_result_rows(result_file)
            except (OSError, ValueError) as error:
                warnings.append(str(error))
                continue

            method_points: dict[int, float] = {}
            for topk in K_VALUES:
                matching = [
                    row
                    for row in rows
                    if int(row["q_batch"]) == q_batch
                    and int(row["topk"]) == topk
                    and int(row["nprobe"]) == dataset.nprobe
                ]
                if not matching:
                    warnings.append(
                        f"{method.label}/{dataset.label}: no QBatch={q_batch}, "
                        f"K={topk}, nprobe={dataset.nprobe} row"
                    )
                    continue
                if len(matching) > 1:
                    qps_values = ", ".join(f'{row["qps"]:.2f}' for row in matching)
                    warnings.append(
                        f"{method.label}/{dataset.label}: {len(matching)} duplicate "
                        f"QBatch={q_batch}, K={topk}, nprobe={dataset.nprobe} rows "
                        f"(QPS={qps_values}); using the latest row"
                    )
                method_points[topk] = matching[-1]["qps"]
            if method_points:
                points[dataset.key][method_key] = method_points
    return points, warnings


def plot_figure(
    points: dict[str, dict[str, dict[int, float]]],
    datasets: tuple[Dataset, ...],
    q_batch: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, len(datasets), figsize=(12.6, 3.8), squeeze=False)
    axes = axes[0]
    legend_handles: dict[str, object] = {}

    for column, dataset in enumerate(datasets):
        axis = axes[column]
        dataset_points = points[dataset.key]
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
                linewidth=2.4,
                markersize=7,
                markeredgewidth=0.7,
                label=method.label,
            )
            legend_handles.setdefault(method_key, line)

        axis.set_xticks(K_VALUES)
        axis.set_xlim(K_VALUES[0] - 5, K_VALUES[-1] + 5)
        axis.set_yscale("log")
        qps_values = [
            qps
            for method_points in dataset_points.values()
            for qps in method_points.values()
        ]
        if qps_values:
            # Keep at least one labeled power-of-ten tick visible in every panel.
            lower_tick = 10 ** math.floor(math.log10(min(qps_values)))
            axis.set_ylim(lower_tick, max(qps_values) * 1.12)
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12)
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("K", fontsize=16)
        axis.set_ylabel("QPS" if column == 0 else "", fontsize=16)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, which="both", linestyle="--", linewidth=0.65, alpha=0.35)
        axis.text(
            0.04,
            0.42,
            dataset.label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
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
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(legend_keys),
            frameon=True,
            fontsize=15,
            handlelength=3.0,
            columnspacing=2.0,
        )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.78, wspace=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    nprobe_tag = "_".join(str(dataset.nprobe) for dataset in datasets)
    stem = f"IVFPQ_SensitiveToK_QBatch{q_batch}_NProbe{nprobe_tag}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-batch", type=int, default=DEFAULT_QBATCH)
    parser.add_argument("--glove-nprobe", type=int, default=300)
    parser.add_argument("--hunyuan-nprobe", type=int, default=700)
    parser.add_argument("--openai-nprobe", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.q_batch <= 0:
        raise ValueError("--q-batch must be positive")
    datasets = (
        Dataset("glove1.2m", "GloVe", args.glove_nprobe),
        Dataset("hunyuan", "Hunyuan", args.hunyuan_nprobe),
        Dataset("openai1536", "OpenAI", args.openai_nprobe),
    )
    if any(dataset.nprobe <= 0 for dataset in datasets):
        raise ValueError("all nprobe values must be positive")

    result_files, discovery_warnings = discover_result_files(DEFAULT_RESULT_DIRS)
    points, collection_warnings = collect_points(result_files, datasets, args.q_batch)
    warnings = discovery_warnings + collection_warnings
    if warnings:
        print("Incomplete or duplicate result entries:")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"Selected QBatch={args.q_batch}")
    for dataset in datasets:
        print(f"{dataset.label}: nprobe={dataset.nprobe}")
        for method_key in LEGEND_ORDER:
            method_points = points[dataset.key].get(method_key)
            if method_points:
                values = ", ".join(
                    f"K={topk}: {method_points[topk]:.2f} QPS"
                    for topk in K_VALUES
                    if topk in method_points
                )
                print(f"  {METHODS[method_key].label}: {values}")

    pdf_path, png_path = plot_figure(points, datasets, args.q_batch, args.output_dir)
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
