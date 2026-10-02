(require :sb-simd)
(in-package :cl-user)
(use-package :sb-simd-avx512f)

(let* ((v (s32.16 -4096))
       (s (s32.16-shiftr v (u32.16 12))))
  (format t "AVX-512 s32.16-shiftr (-4096) >> 12: ~a~%" (aref (s32.16-values s) 0)))
(sb-ext:exit)
