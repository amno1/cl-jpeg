#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBCL_BIN="/home/arthur/repos/sb-simd-512/run-sbcl.sh"

echo "=== Running CL-JPEG AVX-512 Speedup Benchmark on Zen 5 ==="
"${SBCL_BIN}" --dynamic-space-size 4096 \
  --noinform --non-interactive \
  --load "${SCRIPT_DIR}/bench-simd-speedup.lisp"
