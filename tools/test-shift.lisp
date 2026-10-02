(require :sb-simd)
(in-package :cl-user)
(use-package :sb-simd-avx2)

(let* ((v (s32.8 -4096))
       (s (s32.8-shiftr v 12)))
  (format t "v: ~a, shift 12: ~a~%" v s))
(sb-ext:exit)
