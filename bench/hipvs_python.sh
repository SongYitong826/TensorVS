#!/usr/bin/env bash
set -euo pipefail
site=/root/hipvs-venv/lib/python3.12/site-packages
export ROCM_HOME=/opt/rocm
export HIPBLASLT_TENSILE_LIBPATH=/opt/rocm/core-10.1/lib/hipblaslt/library/gfx942
export LD_LIBRARY_PATH="$site/libcuvs/lib64:$site/libraft/lib64:$site/librmm/lib64:$site/rapids_logger/lib64:/opt/rocm/lib:/opt/rocm/core-10.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /root/hipvs-venv/bin/python "$@"
