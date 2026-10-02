(load "~/quicklisp/setup.lisp")
(ql:quickload :flexi-streams :silent t)
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun buffer-hash (vec)
  (let ((hash 5381))
    (dotimes (i (length vec) hash)
      (setf hash (logand (+ (ash hash 5) hash (aref vec i)) #xffffffffffffffff)))))

(multiple-value-bind (buf h w nc)
    (let ((jpeg:*use-simd* :off))
      (jpeg:decode-image #P"/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg"))
  (let* ((bytes-420-point
           (flexi-streams:with-output-to-sequence (s)
             (jpeg:encode-image-stream s buf nc h w :sampling '((2 2)(1 1)(1 1)) :chroma-subsampling :point)))
         (bytes-420-box
           (flexi-streams:with-output-to-sequence (s)
             (jpeg:encode-image-stream s buf nc h w :sampling '((2 2)(1 1)(1 1)) :chroma-subsampling :box)))
         (bytes-444
           (flexi-streams:with-output-to-sequence (s)
             (jpeg:encode-image-stream s buf nc h w :sampling '((1 1)(1 1)(1 1))))))
    (format t "testorig 4:2:0 point: len=~D, hash=#x~16,'0X~%"
            (length bytes-420-point) (buffer-hash bytes-420-point))
    (format t "testorig 4:2:0 box:   len=~D, hash=#x~16,'0X~%"
            (length bytes-420-box) (buffer-hash bytes-420-box))
    (format t "testorig 4:4:4:       len=~D, hash=#x~16,'0X~%"
            (length bytes-444) (buffer-hash bytes-444))))

(sb-ext:exit :code 0)
