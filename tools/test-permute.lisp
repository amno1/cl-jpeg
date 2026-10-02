(require :sb-simd)

(let* ((pkg (find-package :sb-simd-avx2))
       (s32.8-fn (find-symbol "S32.8" pkg))
       (permute-fn (find-symbol "S32.8-PERMUTE" pkg))
       (v (funcall s32.8-fn 10 20 30 40 50 60 70 80))
       (idx (funcall s32.8-fn 7 6 5 4 3 2 1 0)))
  (format t "permute v idx: ~a~%" (funcall permute-fn v idx)))
(sb-ext:exit)
