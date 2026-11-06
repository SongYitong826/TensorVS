"""One transposed table*: datasets across columns; Flat/PQ in two row groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.ivf_resource_table_compact import collect
from plot.ivf_resource_table_example import DATASETS, PLATFORMS, cell, method_label

PLATFORM_LABELS = ("NVIDIA", "Apple", "AMD", "Ascend", "Cambricon", "Kunlunxin")


def make_table(records):
    parts = [r"""\begingroup
\fontsize{6.8}{7.4}\selectfont
\setlength{\tabcolsep}{.8pt}
\setlength{\aboverulesep}{.8pt}
\setlength{\belowrulesep}{.8pt}
\renewcommand{\arraystretch}{.86}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}ll*{18}{r}@{}}
\toprule
\multirow{2}{*}{Platform} & \multirow{2}{*}{Method}
"""]
    parts.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{dataset}}}" for dataset in DATASETS) + r" \\")
    parts.append("".join(rf"\cmidrule(lr){{{3 + 3*i}-{5 + 3*i}}}" for i in range(6)))
    parts.append(" & & " + " & ".join(["B & S & P"] * 6) + r" \\")
    parts.append(r"\midrule")
    for kind_index, kind in enumerate(("ivfflat", "ivfpq")):
        if kind_index:
            parts.append(r"\addlinespace[2pt]")
        parts.append(r"\rowcolor{platformbg}")
        display = "IVF-Flat" if kind == "ivfflat" else "IVF-PQ"
        parts.append(rf"\multicolumn{{20}}{{@{{\hspace{{2pt}}}}l}}{{\strut\textbf{{{display}}}}} \\")
        for platform_index, (_name, supported_datasets, methods) in enumerate(PLATFORMS):
            for method_index, method in enumerate(methods):
                platform = PLATFORM_LABELS[platform_index]
                platform_cell = (rf"\multirow{{{len(methods)}}}{{*}}{{{platform}}}" if len(methods) > 1 else platform) if method_index == 0 else ""
                values = []
                for dataset in DATASETS:
                    if dataset not in supported_datasets:
                        values.append(r"\multicolumn{3}{c}{\textcolor{pending}{---}}")
                        continue
                    entry = records[platform_index, dataset, kind, method]
                    build, size = cell(entry["build"], entry["reused"]), cell(entry["size"])
                    if platform_index == 2 and method_index == 0:
                        build = rf"\multirow{{2}}{{*}}{{{build}}}"
                        size = rf"\multirow{{2}}{{*}}{{{size}}}"
                    elif platform_index == 2 and method_index == 1:
                        build = size = ""
                    values.extend((build, size, cell(entry["peak"])))
                parts.append(" & ".join((platform_cell, method_label(method), *values)) + r" \\")
            if platform_index < len(PLATFORMS) - 1:
                parts.append(r"\cmidrule[.15pt](lr){1-20}")
    parts.append(r"""\bottomrule
\end{tabular*}
\par\vspace{3pt}
{\fontsize{6.3}{7.2}\selectfont
B: build time (s); S: index size (GB); P: peak search memory (GB), $K=10$, QBatch$=128$.
---: outside this platform's dataset coverage; \textit{xxx}: pending; \textsf{N/S}: unsupported; \textsf{CPU}: CPU search.
\textsuperscript{r}Reused source-index build time. AMD TensorVS backends share merged B/S cells.
Peak is maximized over common nprobe values; memory counters and PQ configurations follow each source run.
CUDA values come from the existing table.\par}
\endgroup
""")
    return "\n".join(parts)


def generate(output):
    records = collect()
    colors = r"""\definecolor{platformbg}{HTML}{EAF0F6}
\definecolor{pending}{HTML}{929AA4}
"""
    body = make_table(records)
    preamble = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs,multirow,array}
\usepackage[table]{xcolor}
\begin{document}
"""
    title = r"""{\fontsize{9}{10}\selectfont\textbf{Index and search resources across accelerator platforms.}}\par\vspace{3pt}
"""
    output.write_text(preamble + colors + r"\begin{minipage}{7in}" + "\n" + title + body
                      + r"\end{minipage}\end{document}")
    fragment = ("% Requires booktabs, multirow, array, and xcolor with the table option.\n" + colors
                + r"\begin{table*}[t]" + "\n" + r"\centering" + "\n"
                + r"\caption{Index and search resources across accelerator platforms.}"
                + "\n" + r"\label{tab:ivf-resource-transposed}" + "\n" + body
                + r"\end{table*}" + "\n")
    output.with_suffix(".table.tex").write_text(fragment)
    audit = [dict(platform=PLATFORMS[p][0], dataset=d, kind=k, method=m, **record)
             for (p, d, k, m), record in records.items()]
    output.with_suffix(".sources.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {output}: one continuous matrix, 28 method rows, 80 records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tables" / "ivf_resource_statistics_transposed_example.tex")
    generate(parser.parse_args().output)
