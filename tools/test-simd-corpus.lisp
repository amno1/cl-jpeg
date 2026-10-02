(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(load "tools/test-layout-b.lisp")

(in-package :jpeg)

;; Hook up SIMD DCT
(setf (symbol-function 'llm-dct) #'simd-dct-block-layout-b)

(format t "SIMD DCT hooked to #'simd-dct-block-layout-b.~%")
(format t "Running widened corpus test suite...~%")

(load "tools/test-widened-corpus.lisp")
(sb-ext:exit)
