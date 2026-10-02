#!/usr/bin/env bash
#
# Baseline Benchmark for CL-JPEG
# Measures color conversion, DCT/quantization, and end-to-end codec throughput.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SBCL_BIN="/home/arthur/repos/sb-simd-512/run-sbcl.sh"

"$SBCL_BIN" --noinform --non-interactive \
  --eval "(require :asdf)" \
  --eval "(load (merge-pathnames \"quicklisp/setup.lisp\" (user-homedir-pathname)))" \
  --eval "(push (truename \"$ROOT_DIR\") asdf:*central-registry*)" \
  --eval "(asdf:load-system \"cl-jpeg\")" \
  --load "$SCRIPT_DIR/bench-baseline.lisp" \
  --eval "(sb-ext:exit)"
