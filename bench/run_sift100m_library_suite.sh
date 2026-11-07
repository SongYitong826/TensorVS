#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
library="${1:?Usage: bash bench/run_sift100m_library_suite.sh cuvs|faiss [log-dir]}"
case "$library" in cuvs|faiss) ;; *) echo "Unknown library: $library" >&2; exit 2 ;; esac
log_dir="${2:-result/logs/sift100m_${library}_multigpu_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$log_dir"
if [[ "${3:-}" == --detach ]]; then
    nohup bash "$script_dir/run_sift100m_library_suite.sh" "$library" "$log_dir" \
        > "$log_dir/suite.log" 2>&1 < /dev/null &
    printf 'SUITE_PID=%s\nLOG_DIR=%s\n' "$!" "$log_dir"
    exit 0
fi
printf '%s\n' "$$" > "$log_dir/suite.pid"
printf '%s\n' running > "$log_dir/status"
trap 'printf "%s\n" failed > "$log_dir/status"' ERR
for kind in ivfflat ivfpq; do
    printf '%s\n' "$kind" > "$log_dir/current_method"
    printf '%s\n' "$library $kind: build + query starting $(date -Is)"
    bash bench/run_sift100m_library_bench.sh --library "$library" --kind "$kind" \
        --phase all --gpus 0 1 2 > "$log_dir/$kind.log" 2>&1
    printf '%s\n' "$library $kind: complete $(date -Is)"
done
printf '%s\n' complete > "$log_dir/status"
