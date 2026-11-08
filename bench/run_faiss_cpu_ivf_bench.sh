#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${CONDA_EXE:-$HOME/miniconda3/bin/conda}" run --no-capture-output -n faiss python -u bench/faiss_cpu_ivf_bench.py "$@"
