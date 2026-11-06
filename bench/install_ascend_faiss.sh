#!/usr/bin/env bash
# Deploy to the work disk; run from a server with the tensorvs Conda env.
# Supply the official CANN 9.0.0 toolkit and 910b ops .run installers as arguments.
# Does not run retrieval, training, or benchmark experiments.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${FAISS_ASCEND_RUNTIME:-$(dirname "$project_root")/faiss-ascend-runtime}"
export CONDA_ENVS_PATH="$(dirname "$project_root")/envs${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}"
toolkit="${1:?Usage: bash bench/install_ascend_faiss.sh toolkit.run 910b-ops.run}"
ops="${2:?Supply the matching CANN 9.0.0 910b ops installer}"
mkdir -p "$runtime/logs" "$runtime/wheels" "$runtime/tmp"
export TMPDIR="$runtime/tmp"
if [[ ! -f "$runtime/Ascend/cann-9.0.0/set_env.sh" ]]; then
    conda run --no-capture-output -n tensorvs bash "$toolkit" --full --quiet \
        --install-path="$runtime/Ascend" >"$runtime/logs/cann-toolkit-install.log" 2>&1
fi
set +u
source "$runtime/Ascend/cann-9.0.0/set_env.sh"
set -u
if [[ ! -f "$runtime/910b-ops-installed" ]]; then
    conda run --no-capture-output -n tensorvs bash "$ops" --install --quiet \
        --install-path="$runtime/Ascend" >"$runtime/logs/cann-ops-install.log" 2>&1
    touch "$runtime/910b-ops-installed"
fi
if ! compgen -G "$runtime/wheels/faiss_ascend-1.13.2-26.2.0-*.whl" >/dev/null; then
    conda run --no-capture-output -n tensorvs python -m pip download --no-deps \
        --dest "$runtime/wheels" \
        --index-url https://mirror-centralrepo.devcloud.cn-north-4.huaweicloud.com/artifactory/api/pypi/ascend-pypi/simple \
        faiss-ascend==1.13.2
fi
if [[ ! -f "$runtime/python/faiss/__init__.py" ]]; then
    conda run --no-capture-output -n tensorvs python -m pip install --no-deps \
        --target "$runtime/python" "$runtime"/wheels/faiss_ascend-1.13.2-26.2.0-*.whl
fi
cat >"$runtime/env.sh" <<'ENV'
# Sourced in the Faiss launcher subprocess only.
faiss_runtime="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$faiss_runtime/Ascend/cann-9.0.0/set_env.sh"
export ASCEND_HOME_PATH="$faiss_runtime/Ascend/cann-9.0.0"
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export FAISS_ASCEND_PYTHON="$faiss_runtime/python"
unset FAISS_ASCEND_SOURCE_REVISION FAISS_ASCEND_SOURCE_PATCH_SHA256
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/opp/vendors/faiss_npu_math/op_api/lib:$ASCEND_HOME_PATH/lib64:$(dirname "$faiss_runtime")/envs/tensorvs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/faiss_npu_math"
export PYTHONPATH="$FAISS_ASCEND_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
ENV
set +u
source "$runtime/env.sh"
set -u
# Import installs the embedded custom operators. It does not allocate an index,
# execute queries, train, or benchmark. The stock system toolkit is untouched.
conda run --no-capture-output -n tensorvs python -c \
    'import faiss; print("Faiss Ascend:", faiss.__version__, faiss.__file__); assert hasattr(faiss, "NpuIndexIVFPQ")'
sha256sum "$runtime"/wheels/*.whl >"$runtime/wheels/SHA256SUMS"
echo "Installed Faiss Ascend; no experiments run."
