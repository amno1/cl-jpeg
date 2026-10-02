(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(load "tools/test-layout-b.lisp")

(in-package :jpeg)

(let* ((buf1 (make-array 64 :element-type 'sint16))
       (buf2 (make-array 64 :element-type 'sint16)))
  (dotimes (i 64)
    (let ((v (- (mod (* (1+ i) 1664525) 256) 128)))
      (setf (aref buf1 i) v
            (aref buf2 i) v)))
  (llm-dct buf1 0 1)
  (simd-dct-block-layout-b buf2 0 1)
  ;; Compare [r=3, c=3]
  (format t "Block 0 [r=3, c=3]: scalar=~d, simd=~d~%" (aref buf1 (+ (* 3 8) 3)) (aref buf2 (+ (* 3 8) 3))))

(sb-ext:exit)
