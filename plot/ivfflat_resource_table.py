#!/usr/bin/env python3
"""Generate the IVF-Flat index-resource LaTeX table.

The script uses canonical local result/Build CSV/JSON pairs. Serialized sizes
and build times are separate from historical K=10, QBatch=128 query peaks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.ivf_resource_table import DATASETS, METHODS, TOPK, Q_BATCH, collect_records

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tables" / "ivfflat_resource_statistics.tex"


def highlighted(value: float, minimum: float, maximum: float) -> str:
    rendered = f"{value:.2f}"
    if value == maximum:
        return rf"\cellcolor{{tablemax}}{rendered}"
    if value == minimum:
        return rf"\cellcolor{{tablemin}}{rendered}"
    return rendered


def generate_table() -> str:
    rows: list[str] = []
    for dataset_index, dataset in enumerate(DATASETS):
        records = collect_records(dataset, "IVF-Flat")
        metrics = {
            method.label: {
                "size": records[method.label].size_gib,
                "time": records[method.label].build_seconds,
                "memory": records[method.label].peak_memory_gib,
            }
            for method in METHODS
        }
        extrema = {
            metric: (
                min(values[metric] for values in metrics.values()),
                max(values[metric] for values in metrics.values()),
            )
            for metric in ("size", "time", "memory")
        }

        for method_index, method in enumerate(METHODS):
            prefix = rf"\multirow{{3}}{{*}}{{\texttt{{{dataset.label}}}}}" if method_index == 0 else ""
            values = metrics[method.label]
            cells = [
                highlighted(values[metric], extrema[metric][0], extrema[metric][1])
                for metric in ("size", "time", "memory")
            ]
            rows.append(
                f"    {prefix} & {method.label} & {cells[0]} & {cells[1]} & "
                f"{cells[2]} & -- & -- & -- \\\\" 
            )
        if dataset_index != len(DATASETS) - 1:
            rows.append("    \\midrule")

    body = "\n".join(rows)
    return rf"""\documentclass{{article}}
\usepackage[margin=0.45in]{{geometry}}
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[table]{{xcolor}}
\usepackage{{multirow}}

% Per-dataset maximum and minimum among cuVS, Faiss GPU, and TensorVS.
\definecolor{{tablemax}}{{HTML}}{{FADBD8}} % light red
\definecolor{{tablemin}}{{HTML}}{{D6EAF8}} % light blue

% Replace these three labels when the remaining kernel names are decided.
\newcommand{{\KernelTwo}}{{Kernel 2}}
\newcommand{{\KernelThree}}{{Kernel 3}}
\newcommand{{\KernelFour}}{{Kernel 4}}

\begin{{document}}
\begin{{table*}}[t]
  \centering
  \caption{{IVF-Flat index and GPU resource statistics. Index sizes are
  calculated from the index files. CUDA peak memory is the maximum observed
  over nprobe values shared by all methods with $K={TOPK}$ and
  $\mathrm{{QBatch}}={Q_BATCH}$. Within each dataset
  and metric, \colorbox{{tablemax}}{{maximum}} and
  \colorbox{{tablemin}}{{minimum}} values are highlighted.}}
  \label{{tab:ivfflat-resource-statistics}}
  \setlength{{\tabcolsep}}{{4.2pt}}
  \renewcommand{{\arraystretch}}{{1.10}}
  \resizebox{{\textwidth}}{{!}}{{%
  \begin{{tabular}}{{>{{\centering\arraybackslash}}p{{2.0cm}}lcc*{{4}}{{c}}}}
    \toprule
    \multirow{{2}}{{*}}{{Dataset}} & \multirow{{2}}{{*}}{{Method}}
      & \multirow{{2}}{{*}}{{\shortstack{{Index Size\\(GiB)}}}}
      & \multirow{{2}}{{*}}{{\shortstack{{Build Time\\(s)}}}}
      & \multicolumn{{4}}{{c}}{{Peak GPU Memory (GiB)}} \\
    \cmidrule(lr){{5-8}}
      & & & & CUDA & \KernelTwo & \KernelThree & \KernelFour \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}}}
\end{{table*}}
\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_table())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
