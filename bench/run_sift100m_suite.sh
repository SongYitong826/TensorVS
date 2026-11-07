#!/usr/bin/env bash
# Build/query both methods sequentially; each method uses GPUs 0, 1 and 2.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
log_dir="${1:-result/logs/sift100m_multigpu_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$log_dir"
printf '%s\n' "$$" > "$log_dir/suite.pid"
printf '%s\n' "running" > "$log_dir/status"
trap 'printf "%s\n" "failed" > "$log_dir/status"' ERR
for kind in ivfflat ivfpq; do
    printf '%s\n' "$kind: build + query starting $(date -Is)"
    printf '%s\n' "$kind" > "$log_dir/current_method"
    bash bench/run_sift100m_multigpu_bench.sh \
        --kind "$kind" --phase all --gpus 0 1 2 \
        > "$log_dir/$kind.log" 2>&1
    printf '%s\n' "$kind: complete $(date -Is)"
done
printf '%s\n' "complete" > "$log_dir/status"
printf '%s\n' "All SIFT100M experiments complete $(date -Is)"
