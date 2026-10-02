(in-package :cl-user)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load "~/quicklisp/setup.lisp")
  (asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
  (asdf:load-system :cl-jpeg)
  (require :sb-simd))

(in-package :jpeg)

(load "/home/arthur/repos/cl/cl-jpeg-avx512/tools/proto-idct-simd.lisp")

(defun bench-idct (n-iters)
  (let ((b-orig (make-test-block))
        (b-simd (make-test-block))
        (ws (make-test-block))
        (q (make-array 8 :initial-contents (loop repeat 8 collect (make-array 8 :element-type 'uint8 :initial-element 16)))))
    (dotimes (y 8)
      (dotimes (x 8)
        (setf (aref (aref q y) x) (+ 10 (mod (+ (* y 7) x) 80)))
        (setf (aref (aref b-orig y) x) (- (random 100) 50))
        (setf (aref (aref b-simd y) x) (aref (aref b-orig y) x))))

    ;; Warmup
    (loop repeat 10000 do
      (inverse-llm-dct b-orig q ws)
      (inverse-llm-dct-simd b-simd q ws))

    (format t "Benchmarking ~D iterations of SCALAR inverse-llm-dct...~%" n-iters)
    (let ((t0 (get-internal-real-time)))
      (loop repeat n-iters do
        (inverse-llm-dct b-orig q ws))
      (let ((elapsed (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "SCALAR: ~,3F sec (~,1F ns/block)~%" elapsed (* (/ elapsed n-iters) 1e9))))

    (format t "Benchmarking ~D iterations of SIMD inverse-llm-dct-simd...~%" n-iters)
    (let ((t0 (get-internal-real-time)))
      (loop repeat n-iters do
        (inverse-llm-dct-simd b-simd q ws))
      (let ((elapsed (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "SIMD:   ~,3F sec (~,1F ns/block)~%" elapsed (* (/ elapsed n-iters) 1e9))))))

(bench-idct 500000)
