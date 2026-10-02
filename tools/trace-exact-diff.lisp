(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(let* ((nblocks 8)
       (stride (* nblocks 8))
       (buf1 (make-array (* nblocks 64) :element-type 'sint16)))
  (dotimes (i (* nblocks 64))
    (setf (aref buf1 i) (- (mod (* (1+ i) 1664525) 256) 128)))
  ;; Run standard llm-dct
  (llm-dct buf1 0 nblocks)
  (format t "After standard llm-dct: d3=~d~%" (aref buf1 (+ (* 3 stride) 3))))
(sb-ext:exit)
