(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(load "tools/test-layout-b.lisp")

(in-package :jpeg)

(let* ((nblocks 2)
       (stride (* nblocks 8))
       (buf1 (make-array (* nblocks 64) :element-type 'sint16))
       (buf2 (make-array (* nblocks 64) :element-type 'sint16)))
  (dotimes (i (* nblocks 64))
    (let ((v (- (mod (* (1+ i) 1664525) 256) 128)))
      (setf (aref buf1 i) v
            (aref buf2 i) v)))
  (llm-dct buf1 1 nblocks)
  (simd-dct-block-layout-b buf2 1 nblocks)
  ;; Compare Block 1 [r=0, c=5]
  (let ((idx (+ (* 1 8) (* 0 stride) 5)))
    (format t "Block 1 [r=0, c=5]: scalar=~d, simd=~d~%" (aref buf1 idx) (aref buf2 idx))))

(sb-ext:exit)
