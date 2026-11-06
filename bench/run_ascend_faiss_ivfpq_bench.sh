#!/usr/bin/env bash
# GloVe M32 IVF-PQ: native Faiss Ascend training, build metrics, then retrieval.
set -euo pipefail
if (( $# < 2 )) || [[ "$1" != "ivfpq" || "$2" != "glove1.2m" ]]; then
    echo "Usage: bash bench/run_ascend_faiss_ivfpq_bench.sh ivfpq glove1.2m [options]" >&2
    exit 2
fi
shift 2
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${FAISS_ASCEND_RUNTIME:-$(dirname "$project_root")/faiss-ascend-runtime}"
if [[ -d "$(dirname "$project_root")/envs/tensorvs/conda-meta" ]]; then
    export CONDA_ENVS_PATH="$(dirname "$project_root")/envs${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}"
fi
if [[ ! -f "$runtime/env.sh" ]]; then
    echo "Missing Faiss Ascend deployment: $runtime/env.sh" >&2
    exit 1
fi
# CANN environment scripts may access unset variables.
set +u
source "$runtime/env.sh"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="${FAISS_ASCEND_PYTHON:-$runtime/python}:$project_root/src:$project_root${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n tensorvs python -u \
    "$project_root/bench/ascend_faiss_ivfpq_bench.py" "$@"
