(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(require :sb-simd)

(in-package :jpeg)
(use-package :sb-simd-avx2)

(load "tools/test-layout-b.lisp")

(let* ((nblocks 8)
       (stride (* nblocks 8))
       (buf1 (make-array (* nblocks 64) :element-type 'sint16))
       (buf2 (make-array (* nblocks 64) :element-type 'sint16)))
  (dotimes (i (* nblocks 64))
    (let ((v (- (mod (* (1+ i) 1664525) 256) 128)))
      (setf (aref buf1 i) v
            (aref buf2 i) v)))
  (llm-dct buf1 0 nblocks)
  (simd-dct-block-layout-b buf2 0 nblocks)
  (dotimes (r 8)
    (dotimes (c 8)
      (let* ((idx (+ 0 (* r stride) c))
             (v1 (aref buf1 idx))
             (v2 (aref buf2 idx)))
        (when (/= v1 v2)
          (format t "Diff at [r=~d, c=~d]: scalar=~d, simd=~d~%" r c v1 v2))))))

(sb-ext:exit)
