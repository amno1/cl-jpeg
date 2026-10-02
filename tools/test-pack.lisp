(require :sb-simd)
(in-package :cl-user)
(use-package :sb-simd-avx2)

(defun test-pack ()
  (let* ((r0 (make-s32.8 10 11 12 13 14 15 16 17))
         (r1 (make-s32.8 20 21 22 23 24 25 26 27))
         (packed (s16.16-packs r0 r1))
         ;; vpermq on 64-bit elements: 0xD8 swaps lanes 1 and 2
         (perm (s16.16! (s64.4-permute (s64.4! packed) #xD8)))
         (lo (s16.8-from-s16.16 perm 0))
         (hi (s16.8-from-s16.16 perm 1)))
    (format t "Lo (Row 0): ~a~%" lo)
    (format t "Hi (Row 1): ~a~%" hi)))

(test-pack)
(sb-ext:exit)
