#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
conda_executable="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
exec "$conda_executable" run --no-capture-output -n tensorvs python -u bench/sift100m_library_multigpu_bench.py "$@"
