(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(load "tools/test-layout-b.lisp")

(in-package :jpeg)

(let* ((nblocks 8)
       (stride (* nblocks 8))
       (buf1 (make-array (* nblocks 64) :element-type 'sint16))
       (buf2 (make-array (* nblocks 64) :element-type 'sint16)))
  (dotimes (i (* nblocks 64))
    (let ((v (- (mod (* (1+ i) 1664525) 256) 128)))
      (setf (aref buf1 i) v
            (aref buf2 i) v)))
  (dotimes (b nblocks)
    (llm-dct buf1 b nblocks)
    (simd-dct-block-layout-b buf2 b nblocks))

  (dotimes (b nblocks)
    (let ((b_base (ash b 3)))
      (dotimes (r 8)
        (dotimes (c 8)
          (let* ((idx (+ b_base (* r stride) c))
                 (v1 (aref buf1 idx))
                 (v2 (aref buf2 idx)))
            (when (/= v1 v2)
              (format t "Block ~d [r=~d, c=~d]: scalar=~d, simd=~d, delta=~d~%"
                      b r c v1 v2 (- v2 v1)))))))))

(sb-ext:exit)
