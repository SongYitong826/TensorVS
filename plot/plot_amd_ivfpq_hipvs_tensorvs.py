"""Backward-compatible entry point for the AMD IVF-PQ three-method figure.

Run from the TensorVS project root:
    /root/tensorvs-venv/bin/python plot/plot_amd_ivfpq_hipvs_tensorvs.py
"""

import argparse
from pathlib import Path

from plot_amd_ivf_compare import plot


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=root / "result")
    parser.add_argument("--output-dir", type=Path, default=root / "plot" / "Figures")
    args = parser.parse_args()
    plot(args.result_dir, args.output_dir, "ivfpq")
