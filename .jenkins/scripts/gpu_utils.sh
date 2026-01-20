#!/usr/bin/env bash
set -euo pipefail

killgpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "killgpu: nvidia-smi not found; skipping."
    return 0
  fi

  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | sort -u || true)
  if [ -z "${pids}" ]; then
    echo "killgpu: no GPU PIDs."
    return 0
  fi

  echo "killgpu: killing GPU PIDs -> ${pids}"
  for pid in ${pids}; do
    kill -9 "${pid}" 2>/dev/null || sudo kill -9 "${pid}" 2>/dev/null || echo "killgpu: could not kill ${pid}"
  done
}
