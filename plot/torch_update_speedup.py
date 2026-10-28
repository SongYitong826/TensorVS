"""Plot aggregate same-GPU old/new Torch measurements without keeping old rows.

Run: conda run -n tensorvs python plot/torch_update_speedup.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("sift1m", "glove1.2m", "deep1m", "tiny5m", "gist", "openai1536")
NAMES = ("SIFT1M", "GloVe1.2M", "Deep1M", "Tiny5M", "GIST1M", "OpenAI1536")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "result/torch_update_controlled.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot/Figures")
    args = parser.parse_args()
    report = json.loads(args.source.read_text())
    if report["status"] != "passed" or report["cases"] != 72:
        raise ValueError("Expected the completed 72-case same-GPU comparison")
    rows = {(r["dataset"], r["algorithm"]): r for r in report["results"]}
    values = {kind: [rows[name, kind]["median_qps_speedup"] for name in DATASETS]
              for kind in ("IVFFLAT", "IVFPQ")}
    if any(rows[name, kind]["cases"] != 6 for name, kind in rows):
        raise ValueError("Expected six sampled settings per dataset/index")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(10, 5.3))
    colors = {"IVFFLAT": "#2474B5", "IVFPQ": "#D66A28"}
    for kind, offset, label in (("IVFFLAT", -.17, "IVF-Flat"), ("IVFPQ", .17, "IVF-PQ")):
        positions = [i + offset for i in range(len(DATASETS))]
        axis.barh(positions, values[kind], height=.29, color=colors[kind], label=label)
        for y, value in zip(positions, values[kind]):
            axis.text(value + .018, y, f"{value:.2f}×", va="center", fontsize=10)
    axis.set_yticks(range(len(DATASETS)), NAMES)
    axis.invert_yaxis()
    axis.axvline(1, color="#555555", linestyle="--", linewidth=1.1)
    axis.set_xlim(0, max(1.25, 1.15 * max(value for rates in values.values() for value in rates)))
    axis.set_xlabel("Median QPS ratio (new / old); 1.0 = unchanged")
    axis.set_title("Torch update: same-GPU throughput comparison", pad=14, fontsize=15)
    axis.grid(axis="x", linestyle="--", alpha=.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False)
    figure.text(.5, .035,
                f"RTX 4090 · QBatch=128 · K=10/100 · low/middle/high nprobe · {report['cases']} sampled settings\n"
                f"Same index, queries and runtime; {report['repetitions']} timed sweeps per setting; eager Torch.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, .10, 1, 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "Torch_Update_SameGPU_QPSSpeedup_QBatch128_K10_100"
    outputs = []
    for extension, options in (("png", {"dpi": 180}), ("pdf", {})):
        output = args.output_dir / f"{stem}.{extension}"
        figure.savefig(output, bbox_inches="tight", **options)
        outputs.append(str(output))
    plt.close(figure)
    metadata = dict(source=str(args.source.resolve()),
                    source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    statistic="Median of per-setting new/old QPS ratios, six settings per dataset/index",
                    outputs=outputs)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved same-GPU Torch update summary: PNG + PDF", flush=True)


if __name__ == "__main__":
    main()
