"""Build a compact table* preview: two parallel panels, Flat/PQ side by side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.ivf_resource_table_example import (
    PLATFORMS, amd_records, apple_records, ascend_records, cell, cuda_snapshot, method_label, pending,
)

UNSUPPORTED_CONFIGURATIONS = {
    (0, "Hunyuan", "ivfpq", "Faiss GPU"):
        "Native CUDA IVF-PQ LUT exceeds per-thread-block shared memory for M=384, nbits=8",
    (2, "Hunyuan", "ivfpq", "Faiss Metal"):
        "Metal IVF-PQ wrapper supports M<=64; Hunyuan uses M=384",
    (1, "Hunyuan", "ivfpq", "Faiss ROCm"):
        "ROCm GPU IVF-PQ LUT exceeds per-thread-block shared memory for M=384, nbits=8",
}


def collect():
    snapshot = cuda_snapshot()
    records = {}
    for platform_index, (_platform, datasets, methods) in enumerate(PLATFORMS):
        for dataset in datasets:
            for kind in ("ivfflat", "ivfpq"):
                if platform_index == 0:
                    entries = {method: snapshot[kind, dataset, method] for method in methods}
                elif platform_index == 1:
                    entries = amd_records(dataset, kind)
                elif platform_index == 2:
                    entries = apple_records(dataset, kind)
                elif platform_index == 3:
                    entries = ascend_records(dataset, kind)
                else:
                    entries = {method: pending() for method in methods}
                for method, entry in entries.items():
                    key = platform_index, dataset, kind, method
                    if key in UNSUPPORTED_CONFIGURATIONS:
                        # The table reports accelerator support; CPU measurements
                        # remain intact in their original result/Build files.
                        entry = dict(entry, build="N/S", size="N/S", peak="N/S",
                                     reused=False, unsupported_reason=UNSUPPORTED_CONFIGURATIONS[key])
                    records[key] = entry
    assert len(records) == 76
    return records


def panel(platform_indexes, records):
    parts = [r"""\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrr@{}}
\toprule
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Method}
 & \multicolumn{3}{c}{IVF-Flat} & \multicolumn{3}{c}{IVF-PQ} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
 & & Build & Size & Peak & Build & Size & Peak \\
\midrule
"""]
    for group_index, platform_index in enumerate(platform_indexes):
        platform, datasets, methods = PLATFORMS[platform_index]
        if group_index:
            parts.append(r"\addlinespace[2pt]")
        parts.append(r"\rowcolor{platformbg}")
        parts.append(rf"\multicolumn{{8}}{{@{{\hspace{{2pt}}}}l}}{{\strut\textbf{{{platform}}}}} \\")
        for dataset_index, dataset in enumerate(datasets):
            for method_index, method in enumerate(methods):
                dataset_cell = (rf"\multirow{{{len(methods)}}}{{*}}{{{dataset}}}" if len(methods) > 1 else dataset) if method_index == 0 else ""
                values = []
                for kind in ("ivfflat", "ivfpq"):
                    entry = records[platform_index, dataset, kind, method]
                    build = cell(entry["build"], entry["reused"])
                    size = cell(entry["size"])
                    if platform_index == 1 and method_index == 0:
                        build = rf"\multirow{{2}}{{*}}{{{build}}}"
                        size = rf"\multirow{{2}}{{*}}{{{size}}}"
                    elif platform_index == 1 and method_index == 1:
                        build = size = ""
                    values.extend((build, size, cell(entry["peak"])))
                parts.append(" & ".join((dataset_cell, method_label(method), *values)) + r" \\")
            if dataset_index != len(datasets) - 1:
                parts.append(r"\cmidrule[.15pt](lr){1-8}")
    parts.append(r"\bottomrule\end{tabular*}")
    return "\n".join(parts)


def generate(output):
    records = collect()
    # Balance height: 22 method rows on the left, 18 plus more platform bands on the right.
    left, right = panel((0, 1), records), panel((2, 3, 4, 5), records)
    colors = r"""\definecolor{platformbg}{HTML}{EAF0F6}
\definecolor{pending}{HTML}{929AA4}
"""
    body = r"""\begingroup
\fontsize{7}{8.1}\selectfont
\setlength{\tabcolsep}{1.1pt}
\setlength{\aboverulesep}{1.0pt}
\setlength{\belowrulesep}{1.0pt}
\renewcommand{\arraystretch}{.86}
\begin{minipage}[t]{.489\linewidth}
\vspace{0pt}
""" + left + r"""
\end{minipage}\hfill
\begin{minipage}[t]{.489\linewidth}
\vspace{0pt}
""" + right + r"""
\end{minipage}
\par\vspace{3pt}
{\fontsize{6.3}{7.3}\selectfont
Build: s; Size/Peak: GB. Peak: search at $K=10$, QBatch$=128$, maximized over common nprobe settings.
\textit{xxx}: pending; \textsf{N/S}: unsupported; \textsf{CPU}: CPU search.
\textsuperscript{r}Reused source-index build time.
Memory counters and PQ configurations follow each source run; compare within a platform. CUDA uses canonical Build records and historical query-memory summaries.\par}
\endgroup
"""
    preamble = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs,multirow,array}
\usepackage[table]{xcolor}
\begin{document}
"""
    title = r"""{\fontsize{9}{10}\selectfont\textbf{Resource costs across accelerator platforms.}}\par\vspace{3pt}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(preamble + colors + r"\noindent\begin{minipage}{7in}" + "\n" + title + body
                      + "\n" + r"\end{minipage}\end{document}")
    # This fragment can be pasted into the two-column paper without scaling a full page.
    fragment = ("% Requires booktabs, multirow, array, and xcolor with the table option.\n"
                + colors + "\n" + r"\begin{table*}[t]" + "\n" + r"\centering" + "\n"
                + r"\caption{Index build time, index size, and peak search memory across accelerator platforms.}"
                + "\n" + r"\label{tab:ivf-resource-multiplatform}" + "\n" + body + r"\end{table*}" + "\n")
    output.with_suffix(".table.tex").write_text(fragment)
    audit = [dict(platform=PLATFORMS[platform][0], dataset=dataset, kind=kind, method=method, **entry)
             for (platform, dataset, kind, method), entry in records.items()]
    output.with_suffix(".sources.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {output}; 80 records in one two-panel table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tables" / "ivf_resource_statistics_compact_example.tex")
    generate(parser.parse_args().output)
