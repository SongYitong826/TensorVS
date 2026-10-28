"""Plot the 3x6 IVFFlat CUDA-kernel QPS/Recall figure.

The rows are query batch sizes 32, 128, and 512.  The columns are the six
paper datasets in their fixed display order.  TensorVS is compared with cuVS
and Faiss GPU at K=10 by default.  Missing results are kept as explicit
``Results pending`` panels so that the final layout does not change when new
CSV files are added.

Run from the TensorVS repository root with::

    conda run -n tensorvs python plot/ivfflat_cuda_kernel.py

The script discovers files named ``IVFFLAT_<dataset>_nlist*.csv`` in each
implementation's result directory and writes both PDF and PNG figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plot" / "Figures"


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str


@dataclass(frozen=True)
class Method:
    label: str
    color: str
    marker: str
    linestyle: str


DATASETS = (
    Dataset("sift10m", "SIFT"),
    Dataset("glove1.2m", "GloVe"),
    Dataset("tiny5m", "Tiny"),
    Dataset("hunyuan", "Hunyuan"),
    Dataset("gist", "GIST"),
    Dataset("openai1536", "OpenAI"),
)
QBATCHES = (32, 128, 512)

METHODS = {
    "tensorvs": Method("TensorVS", "#1f77b4", "o", "-"),
    "cuvs": Method("cuVS", "#ff7f0e", "s", "--"),
    "faiss": Method("Faiss GPU", "#2ca02c", "^", "-."),
}
LEGEND_ORDER = ("cuvs", "faiss", "tensorvs")
DATASET_Y_LIMITS = {
    # The Hunyuan methods form two separated QPS bands.  A fixed two-decade
    # range keeps the shared scale readable without letting the central gap
    # dominate each panel.
    "hunyuan": (1e2, 1e4),
}

DEFAULT_RESULT_DIRS = {
    "tensorvs": (PROJECT_ROOT / "result", PROJECT_ROOT / "results"),
    "cuvs": (
        Path("/home/songyitong/cuvs/python/cuvs/cuvs/result"),
        Path("/home/songyitong/cuvs/python/cuvs/cuvs/results"),
    ),
    "faiss": (
        Path("/home/songyitong/faiss/result"),
        Path("/home/songyitong/faiss/results"),
    ),
}

FILENAME_PATTERN = re.compile(
    r"^IVFFLAT_(?P<dataset>.+?)_nlist(?P<nlist>\d+)"
    r"(?:_K(?P<topk>\d+))?\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResultFile:
    path: Path
    filename_topk: int | None


def discover_result_files(
    result_dirs: dict[str, tuple[Path, ...]],
) -> dict[str, dict[str, ResultFile]]:
    """Find one IVFFlat CSV per method and requested dataset."""

    requested = {dataset.key.casefold() for dataset in DATASETS}
    discovered: dict[str, dict[str, ResultFile]] = {method: {} for method in METHODS}
    for method in METHODS:
        for result_dir in result_dirs[method]:
            if not result_dir.is_dir():
                continue
            for path in sorted(result_dir.glob("*.csv")):
                match = FILENAME_PATTERN.match(path.name)
                if match is None:
                    continue
                dataset = match.group("dataset").casefold()
                if dataset not in requested:
                    continue
                if dataset in discovered[method]:
                    previous = discovered[method][dataset].path
                    print(
                        f"Warning: ignoring duplicate {METHODS[method].label}/"
                        f"{dataset} result {path}; using {previous}"
                    )
                    continue
                filename_topk = match.group("topk")
                discovered[method][dataset] = ResultFile(
                    path=path,
                    filename_topk=int(filename_topk) if filename_topk else None,
                )
    return discovered


def _column_name(fieldnames: list[str], *accepted: str) -> str | None:
    by_casefold = {name.casefold(): name for name in fieldnames}
    return next((by_casefold[name.casefold()] for name in accepted if name.casefold() in by_casefold), None)


def read_result_rows(result_file: ResultFile) -> list[dict[str, float]]:
    """Read the fields needed by the figure from one benchmark CSV."""

    with result_file.path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        q_batch_column = _column_name(fieldnames, "q_batch", "qbatch")
        topk_column = _column_name(fieldnames, "K", "topk")
        qps_column = _column_name(fieldnames, "qps")
        recall_column = _column_name(fieldnames, "recall")
        missing = [
            label
            for label, column in (
                ("q_batch", q_batch_column),
                ("qps", qps_column),
                ("recall", recall_column),
            )
            if column is None
        ]
        if topk_column is None and result_file.filename_topk is None:
            missing.append("K")
        if missing:
            raise ValueError(
                f"{result_file.path} is missing columns: {', '.join(missing)}"
            )

        rows: list[dict[str, float]] = []
        for csv_row in reader:
            try:
                row = {
                    "q_batch": float(csv_row[q_batch_column]),
                    "topk": (
                        float(csv_row[topk_column])
                        if topk_column is not None
                        else float(result_file.filename_topk)
                    ),
                    "qps": float(csv_row[qps_column]),
                    "recall": float(csv_row[recall_column]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in row.values()) and row["qps"] > 0:
                rows.append(row)
    return rows


def clip_recall_curve(
    rows: list[dict[str, float]], recall_min: float, recall_max: float
) -> list[dict[str, float]]:
    """Clip a curve to the displayed recall range, interpolating its left edge."""

    rows = sorted(rows, key=lambda row: row["recall"])
    selected = [row for row in rows if recall_min <= row["recall"] <= recall_max]

    lower = next((row for row in reversed(rows) if row["recall"] < recall_min), None)
    upper = next((row for row in rows if row["recall"] > recall_min), None)
    has_boundary = any(
        math.isclose(row["recall"], recall_min, rel_tol=0.0, abs_tol=1e-12)
        for row in rows
    )
    if not has_boundary and lower is not None and upper is not None:
        ratio = (recall_min - lower["recall"]) / (upper["recall"] - lower["recall"])
        selected.append(
            {
                "recall": recall_min,
                "qps": lower["qps"] + ratio * (upper["qps"] - lower["qps"]),
                "q_batch": lower["q_batch"],
                "topk": lower["topk"],
            }
        )
    selected.sort(key=lambda row: row["recall"])
    return selected


def collect_curves(
    result_files: dict[str, dict[str, ResultFile]],
    q_batches: tuple[int, ...],
    topk: int,
    recall_min: float,
    recall_max: float,
) -> tuple[
    dict[str, dict[int, dict[str, list[dict[str, float]]]]],
    list[str],
]:
    """Load and group curves as dataset -> QBatch -> method -> points."""

    curves: dict[str, dict[int, dict[str, list[dict[str, float]]]]] = {
        dataset.key: {q_batch: {} for q_batch in q_batches} for dataset in DATASETS
    }
    warnings: list[str] = []
    for method in METHODS:
        for dataset in DATASETS:
            result_file = result_files[method].get(dataset.key)
            if result_file is None:
                warnings.append(f"{METHODS[method].label}/{dataset.label}: file missing")
                continue
            try:
                rows = read_result_rows(result_file)
            except (OSError, ValueError) as error:
                warnings.append(str(error))
                continue

            for q_batch in q_batches:
                matching = [
                    row
                    for row in rows
                    if int(row["q_batch"]) == q_batch and int(row["topk"]) == topk
                ]
                curve = clip_recall_curve(matching, recall_min, recall_max)
                if curve:
                    curves[dataset.key][q_batch][method] = curve
                else:
                    warnings.append(
                        f"{METHODS[method].label}/{dataset.label}: no visible "
                        f"QBatch={q_batch}, K={topk} points"
                    )
    return curves, warnings


def _dataset_y_limits(
    dataset_curves: dict[int, dict[str, list[dict[str, float]]]],
) -> tuple[float, float] | None:
    values = [
        row["qps"]
        for method_curves in dataset_curves.values()
        for curve in method_curves.values()
        for row in curve
    ]
    if not values:
        return None
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return low / 1.5, high * 1.5
    return low / 1.25, high * 1.25


def plot_figure(
    curves: dict[str, dict[int, dict[str, list[dict[str, float]]]]],
    q_batches: tuple[int, ...],
    topk: int,
    recall_min: float,
    recall_max: float,
    output_dir: Path,
    output_stem: str,
) -> tuple[Path, Path]:
    """Render and save the fixed-layout paper figure."""

    figure, axes = plt.subplots(
        len(q_batches),
        len(DATASETS),
        figsize=(18.0, 8.0),
        squeeze=False,
    )
    legend_handles: dict[str, object] = {}

    for column, dataset in enumerate(DATASETS):
        y_limits = DATASET_Y_LIMITS.get(
            dataset.key, _dataset_y_limits(curves[dataset.key])
        )
        for row_index, q_batch in enumerate(q_batches):
            axis = axes[row_index][column]
            method_curves = curves[dataset.key][q_batch]
            for method_key, method in METHODS.items():
                curve = method_curves.get(method_key)
                if not curve:
                    continue
                (line,) = axis.plot(
                    [point["recall"] for point in curve],
                    [point["qps"] for point in curve],
                    color=method.color,
                    marker=method.marker,
                    linestyle=method.linestyle,
                    linewidth=2.0,
                    markersize=4.5,
                    markeredgewidth=0.6,
                    label=method.label,
                )
                legend_handles.setdefault(method_key, line)

            axis.set_xlim(recall_min, recall_max)
            axis.set_xticks(
                [
                    recall_min + index * (recall_max - recall_min) / 5
                    for index in range(6)
                ]
            )
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
            axis.yaxis.set_minor_locator(
                LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12)
            )
            axis.yaxis.set_minor_formatter(NullFormatter())
            if y_limits is not None:
                axis.set_ylim(*y_limits)
            axis.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
            axis.tick_params(axis="both", labelsize=12.5)
            axis.set_xlabel("Recall", fontsize=14)
            axis.set_ylabel("QPS" if column == 0 else "", fontsize=14)
            axis.text(
                0.04,
                0.06,
                dataset.label,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=13.5,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
            axis.text(
                0.96,
                0.94,
                f"QBatch = {q_batch}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=12.5,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )

            if not method_curves:
                axis.text(
                    0.5,
                    0.5,
                    "Results pending",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="#777777",
                )

    legend_method_keys = [key for key in LEGEND_ORDER if key in legend_handles]
    if legend_method_keys:
        figure.legend(
            [legend_handles[key] for key in legend_method_keys],
            [METHODS[key].label for key in legend_method_keys],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_method_keys),
            frameon=True,
            fontsize=16,
            handlelength=3.0,
            columnspacing=2.2,
        )
    figure.subplots_adjust(
        left=0.052,
        right=0.995,
        bottom=0.09,
        top=0.91,
        wspace=0.27,
        hspace=0.32,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def _selected_dirs(
    supplied: list[Path] | None, defaults: tuple[Path, ...]
) -> tuple[Path, ...]:
    return tuple(supplied) if supplied else defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the 3x6 IVFFlat CUDA-kernel QPS/Recall figure."
    )
    parser.add_argument("--k", type=int, default=10, help="Top-K value (default: 10).")
    parser.add_argument(
        "--q-batches",
        type=int,
        nargs="+",
        default=list(QBATCHES),
        help="Row query batch sizes (default: 32 128 512).",
    )
    parser.add_argument("--recall-min", type=float, default=0.9)
    parser.add_argument("--recall-max", type=float, default=1.0)
    parser.add_argument(
        "--tensorvs-dir",
        action="append",
        type=Path,
        help="TensorVS result directory; repeat to search multiple directories.",
    )
    parser.add_argument(
        "--cuvs-dir",
        action="append",
        type=Path,
        help="cuVS result directory; repeat to search multiple directories.",
    )
    parser.add_argument(
        "--faiss-dir",
        action="append",
        type=Path,
        help="Faiss GPU result directory; repeat to search multiple directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output filename without extension; a parameterized name is used by default.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail instead of drawing the figure if any requested result is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.recall_min < args.recall_max <= 1.0:
        raise ValueError("Recall bounds must satisfy 0 <= min < max <= 1.")
    q_batches = tuple(args.q_batches)
    result_dirs = {
        "tensorvs": _selected_dirs(args.tensorvs_dir, DEFAULT_RESULT_DIRS["tensorvs"]),
        "cuvs": _selected_dirs(args.cuvs_dir, DEFAULT_RESULT_DIRS["cuvs"]),
        "faiss": _selected_dirs(args.faiss_dir, DEFAULT_RESULT_DIRS["faiss"]),
    }

    result_files = discover_result_files(result_dirs)
    curves, warnings = collect_curves(
        result_files,
        q_batches=q_batches,
        topk=args.k,
        recall_min=args.recall_min,
        recall_max=args.recall_max,
    )
    if warnings:
        print("Incomplete result entries:")
        for warning in warnings:
            print(f"  - {warning}")
    if args.require_complete and warnings:
        raise RuntimeError(f"{len(warnings)} requested result entries are incomplete")

    output_stem = args.output_stem or (
        f"IVFFlat_CUDA_QBatch{'_'.join(map(str, q_batches))}_K{args.k}"
    )
    pdf_path, png_path = plot_figure(
        curves,
        q_batches=q_batches,
        topk=args.k,
        recall_min=args.recall_min,
        recall_max=args.recall_max,
        output_dir=args.output_dir,
        output_stem=output_stem,
    )
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
