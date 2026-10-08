"""Plot IVF-Flat/IVF-PQ QPS as K changes.

Each figure contains one row of dataset subplots ordered by vector dimension.
The plot keeps ``QBatch`` and ``nprobe`` fixed so that the horizontal axis is
the only changing search parameter.

Run from the project root with::

    conda run -n tensorvs python plot/ivf_qps_k_plot.py

The default is IVF-Flat with ``QBatch=128``.  Without ``--nprobe``, one figure
is generated for every available nprobe value.  Examples::

    python ivf_qps_k_plot.py --nprobe 120
    python ivf_qps_k_plot.py --index-type ivfpq --q-batch 128 --nprobe 120
    python ivf_qps_k_plot.py --index-type both --nprobe 120
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

try:
    from .ivfflat_plot import (
        AXIS_LABEL_FONTSIZE,
        DATASET_LABEL_FONTSIZE,
        DATASET_DIMENSIONS,
        DEFAULT_OUTPUT_DIR,
        FIGURE_HEIGHT,
        FIGURE_WIDTH_PER_DATASET,
        LEGEND_FONTSIZE,
        LINE_WIDTH,
        MARKER_SIZE,
        RESULT_DIRS,
        STYLES,
        TICK_LABEL_FONTSIZE,
        dataset_display_name,
        dataset_sort_key,
    )
except ImportError:  # Direct execution: ``python plot/ivf_qps_k_plot.py``.
    from ivfflat_plot import (
        AXIS_LABEL_FONTSIZE,
        DATASET_LABEL_FONTSIZE,
        DATASET_DIMENSIONS,
        DEFAULT_OUTPUT_DIR,
        FIGURE_HEIGHT,
        FIGURE_WIDTH_PER_DATASET,
        LEGEND_FONTSIZE,
        LINE_WIDTH,
        MARKER_SIZE,
        RESULT_DIRS,
        STYLES,
        TICK_LABEL_FONTSIZE,
        dataset_display_name,
        dataset_sort_key,
    )


INDEX_PREFIXES = {
    "ivfflat": "IVFFLAT_",
    "ivfpq": "IVFPQ_",
}

# Extra room above the fastest curve on the logarithmic QPS axis. Besides
# preventing the top curve from touching the frame, the larger displayed
# range makes the separation between high- and low-QPS curves less dominant.
Y_AXIS_TOP_HEADROOM_FACTOR = 1.6
Y_AXIS_BOTTOM_MARGIN_FACTOR = 0.85
RECALL_LABEL_FONTSIZE = 12
RECALL_LABEL_VERTICAL_OFFSETS = {
    "TensorVS": 10,
    "cuVS": 0,
    "Faiss": -10,
}
RECALL_LABEL_HORIZONTAL_OFFSETS = {
    # Keep the high-QPS annotation away from the dataset name in the upper
    # right, while leaving the lower annotations close to their endpoints.
    "TensorVS": -48,
    "cuVS": -6,
    "Faiss": -6,
}
# The Recall ranges of these datasets are close enough that the default
# method-only offsets make neighboring annotations overlap. Keep the labels
# attached to their endpoint, but give the affected panels independent
# vertical placements.
RECALL_LABEL_VERTICAL_OFFSETS_BY_DATASET = {
    "deep1m": {
        "TensorVS": 10,
        "cuVS": 24,
        "Faiss": -24,
    },
    "glove1.2m": {
        "TensorVS": 10,
        "cuVS": -20,
        "Faiss": 18,
    },
    "tiny5m": {
        "TensorVS": -18,
        "cuVS": 16,
        "Faiss": -10,
    },
    "gist": {
        "TensorVS": 10,
        "cuVS": 16,
        "Faiss": -18,
    },
}

@dataclass(frozen=True)
class ResultFile:
    """One result CSV and the metadata encoded in its filename."""

    method: str
    index_type: str
    dataset: str
    path: Path
    filename_k: int | None


def result_metadata_from_filename(path: Path) -> tuple[str, str, int | None] | None:
    """Extract ``(index_type, dataset, legacy K)`` from a result filename."""

    for index_type, prefix in INDEX_PREFIXES.items():
        if not path.stem.upper().startswith(prefix):
            continue

        dataset_and_config = path.stem[len(prefix) :]
        topk_match = re.search(r"_K(?P<topk>\d+)$", dataset_and_config, re.IGNORECASE)
        if topk_match is None:
            dataset_and_nlist = dataset_and_config
            filename_k = None
        else:
            dataset_and_nlist = dataset_and_config[: topk_match.start()]
            filename_k = int(topk_match.group("topk"))

        dataset = dataset_and_nlist.split("_nlist", maxsplit=1)[0]
        if dataset:
            return index_type, dataset, filename_k
        return None

    return None


def discover_result_files(index_type_filter: str) -> list[ResultFile]:
    """Find result files for the selected index type and benchmark methods."""

    discovered: dict[tuple[str, str, str], ResultFile] = {}
    selected_types = set(INDEX_PREFIXES) if index_type_filter == "both" else {index_type_filter}
    for method, result_dir in RESULT_DIRS.items():
        if not result_dir.is_dir():
            continue

        for path in sorted(result_dir.glob("*.csv")):
            metadata = result_metadata_from_filename(path)
            if metadata is None:
                continue
            index_type, dataset, filename_k = metadata
            if index_type not in selected_types:
                continue
            key = (method, index_type, dataset)
            if key in discovered:
                print(
                    f"Warning: multiple {method}/{index_type}/{dataset} files; "
                    f"using {discovered[key].path}"
                )
                continue
            discovered[key] = ResultFile(
                method=method,
                index_type=index_type,
                dataset=dataset,
                path=path,
                filename_k=filename_k,
            )
    return list(discovered.values())


def read_results(result_file: ResultFile) -> list[dict[str, float]]:
    """Read valid ``(QBatch, nprobe, K, QPS, Recall)`` rows from one CSV."""

    with result_file.path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"nprobes", "q_batch", "qps", "recall"}
        actual_columns = set(reader.fieldnames or ())
        missing = required_columns - actual_columns
        if missing:
            raise ValueError(
                f"{result_file.path} is missing columns: {', '.join(sorted(missing))}"
            )

        k_column = next(
            (column for column in (reader.fieldnames or ()) if column.casefold() == "k"),
            None,
        )
        if k_column is None and result_file.filename_k is None:
            raise ValueError(f"{result_file.path} is missing the K column")

        rows = []
        for row in reader:
            rows.append(
                {
                    "nprobe": float(row["nprobes"]),
                    "q_batch": float(row["q_batch"]),
                    "K": float(row[k_column]) if k_column is not None else float(result_file.filename_k),
                    "qps": float(row["qps"]),
                    "recall": float(row["recall"]),
                }
            )

    return [row for row in rows if row["qps"] > 0]


def collect_groups(
    result_files: list[ResultFile],
    q_batch_filter: int | None,
    nprobe_filter: int | None,
) -> dict[tuple[str, int, int], dict[str, dict[str, dict[int, dict[str, float]]]]]:
    """Group rows by index type, QBatch, nprobe, dataset, and method."""

    groups: dict[
        tuple[str, int, int], dict[str, dict[str, dict[int, dict[str, float]]]]
    ] = {}
    for result_file in result_files:
        try:
            rows = read_results(result_file)
        except (OSError, ValueError) as error:
            print(f"Warning: skipping {result_file.path}: {error}")
            continue

        for row in rows:
            q_batch = int(row["q_batch"])
            nprobe = int(row["nprobe"])
            if q_batch_filter is not None and q_batch != q_batch_filter:
                continue
            if nprobe_filter is not None and nprobe != nprobe_filter:
                continue

            group = groups.setdefault(
                (result_file.index_type, q_batch, nprobe), {}
            )
            dataset_group = group.setdefault(result_file.dataset, {})
            method_points = dataset_group.setdefault(result_file.method, {})
            k = int(row["K"])
            if k in method_points:
                print(
                    f"Warning: duplicate {result_file.method}/{result_file.index_type}/"
                    f"{result_file.dataset}/QBatch{q_batch}/nprobe{nprobe}/K{k}; "
                    "keeping the first row"
                )
                continue
            method_points[k] = {
                "qps": row["qps"],
                "recall": row["recall"],
            }

    return groups


def plot_group(
    index_type: str,
    q_batch: int,
    nprobe: int,
    grouped_results: dict[str, dict[str, dict[int, dict[str, float]]]],
    output_dir: Path,
) -> Path | None:
    """Plot one index/QBatch/nprobe group as one row of dataset subplots."""

    dataset_results = {
        dataset: method_points
        for dataset, method_points in grouped_results.items()
        if any(method_points.values())
    }
    if not dataset_results:
        return None

    datasets = sorted(dataset_results, key=dataset_sort_key)
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(FIGURE_WIDTH_PER_DATASET * len(datasets), FIGURE_HEIGHT),
        squeeze=False,
    )
    axes = axes[0]
    legend_handles: dict[str, object] = {}

    for axis, dataset in zip(axes, datasets):
        recall_labels: list[tuple[str, float, float, str]] = []
        for method in RESULT_DIRS:
            points = dataset_results[dataset].get(method)
            if not points:
                continue
            style = STYLES[method]
            k_values = sorted(points)
            (line,) = axis.plot(
                k_values,
                [points[k]["qps"] for k in k_values],
                label=method,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                markeredgewidth=0.8,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                **style,
            )
            legend_handles.setdefault(method, line)
            recall_values = [
                points[k]["recall"] for k in k_values if 10 <= k <= 100
            ]
            if recall_values:
                recall_labels.append(
                    (
                        method,
                        min(recall_values),
                        max(recall_values),
                        style["color"],
                        k_values[-1],
                        points[k_values[-1]]["qps"],
                    )
                )

        all_k_values = sorted(
            {
                k
                for method_points in dataset_results[dataset].values()
                for k in method_points
            }
        )
        axis.set_xlabel("K", fontsize=AXIS_LABEL_FONTSIZE)
        axis.set_ylabel("QPS", fontsize=AXIS_LABEL_FONTSIZE)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        axis.set_xticks(all_k_values)
        if len(all_k_values) == 1:
            axis.set_xlim(all_k_values[0] - 1, all_k_values[0] + 1)
        else:
            padding = max(1, math.ceil((all_k_values[-1] - all_k_values[0]) * 0.05))
            axis.set_xlim(all_k_values[0] - padding, all_k_values[-1] + padding)
        axis.set_yscale("log")
        qps_values = [
            qps
            for method_points in dataset_results[dataset].values()
            for point in method_points.values()
            for qps in (point["qps"],)
        ]
        axis.set_ylim(
            min(qps_values) * Y_AXIS_BOTTOM_MARGIN_FACTOR,
            max(qps_values) * Y_AXIS_TOP_HEADROOM_FACTOR,
        )
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.text(
            0.97,
            0.87,
            dataset_display_name(dataset),
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=DATASET_LABEL_FONTSIZE + 3,
            clip_on=False,
        )
        for method, recall_min, recall_max, color, last_k, last_qps in recall_labels:
            axis.annotate(
                f"R: {recall_min:.4f}–{recall_max:.4f}",
                xy=(last_k, last_qps),
                xytext=(
                    RECALL_LABEL_HORIZONTAL_OFFSETS.get(method, -6),
                    RECALL_LABEL_VERTICAL_OFFSETS_BY_DATASET.get(
                        dataset,
                        RECALL_LABEL_VERTICAL_OFFSETS,
                    ).get(
                        method,
                        RECALL_LABEL_VERTICAL_OFFSETS.get(method, 0),
                    ),
                ),
                textcoords="offset points",
                ha="right",
                va="center",
                color=color,
                fontsize=RECALL_LABEL_FONTSIZE,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 2.0,
                },
                zorder=5,
                annotation_clip=False,
            )

    legend_methods = [method for method in RESULT_DIRS if method in legend_handles]
    figure.legend(
        [legend_handles[method] for method in legend_methods],
        legend_methods,
        loc="lower center",
        # The subplot row starts after the left margin reserved for the
        # repeated QPS labels, so its visual center is slightly right of the
        # figure center.
        bbox_to_anchor=(0.58, 0.90),
        bbox_transform=figure.transFigure,
        borderaxespad=0.1,
        ncol=len(legend_methods),
        frameon=True,
        fontsize=LEGEND_FONTSIZE + 3,
        handlelength=2.4,
        columnspacing=1.5,
    )
    figure.tight_layout(rect=(0.16, 0, 1, 0.87), w_pad=2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{index_type.upper()}_QPSvsK_QBatch{q_batch}_NProbe{nprobe}.pdf"
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to: {output}")
    return output


def plot_all(
    index_type_filter: str,
    q_batch_filter: int | None,
    nprobe_filter: int | None,
    output_dir: Path,
) -> list[Path]:
    """Generate QPS-vs-K figures for all matching parameter groups."""

    result_files = discover_result_files(index_type_filter)
    groups = collect_groups(result_files, q_batch_filter, nprobe_filter)
    outputs = []
    for (index_type, q_batch, nprobe), grouped_results in sorted(groups.items()):
        output = plot_group(
            index_type=index_type,
            q_batch=q_batch,
            nprobe=nprobe,
            grouped_results=grouped_results,
            output_dir=output_dir,
        )
        if output is not None:
            outputs.append(output)

    if not outputs:
        filters = [f"index type={index_type_filter}"]
        if q_batch_filter is not None:
            filters.append(f"QBatch={q_batch_filter}")
        if nprobe_filter is not None:
            filters.append(f"nprobe={nprobe_filter}")
        raise ValueError(f"No valid IVF QPS-vs-K plots found for {', '.join(filters)}.")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot IVF-Flat/IVF-PQ QPS as K changes."
    )
    parser.add_argument(
        "--index-type",
        choices=("ivfflat", "ivfpq", "both"),
        default="ivfflat",
        help="Index type to plot; default: ivfflat.",
    )
    parser.add_argument(
        "--q-batch",
        type=int,
        default=128,
        help="Only generate this QBatch figure; default: 128.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="Only generate this nprobe figure; default: generate all available nprobe values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_all(
        index_type_filter=args.index_type,
        q_batch_filter=args.q_batch,
        nprobe_filter=args.nprobe,
        output_dir=args.output_dir,
    )
