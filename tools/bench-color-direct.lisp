(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun bench-color-direct (w h reps)
  (let* ((desc (make-descriptor))
         (buf (make-array (* w h 3) :element-type 'uint8 :initial-element 128)))
    (setf (descriptor-width desc) w
          (descriptor-height desc) h
          (descriptor-ncomp desc) 3
          (descriptor-buffer desc) buf)
    ;; Warmup
    (dotimes (i 5)
      (inverse-colorspace-convert desc)
      (inverse-colorspace-convert-simd-accelerated desc))
    ;; Measure scalar
    (let ((best-scal 1e9))
      (dotimes (b 5)
        (sb-ext:gc :full t)
        (let ((t0 (get-internal-real-time)))
          (dotimes (r reps) (inverse-colorspace-convert desc))
          (let ((ms (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0)))
            (when (< ms best-scal) (setf best-scal ms)))))
      ;; Measure SIMD
      (let ((best-simd 1e9))
        (dotimes (b 5)
          (sb-ext:gc :full t)
          (let ((t0 (get-internal-real-time)))
            (dotimes (r reps) (inverse-colorspace-convert-simd-accelerated desc))
            (let ((ms (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0)))
              (when (< ms best-simd) (setf best-simd ms)))))
        (format t "Isolated Color Conversion (~Dx~D, ~D reps):~%" w h reps)
        (format t "  Scalar Color: ~6,3F ms (~6,1F Mpix/s)~%" best-scal (/ (* w h 1e-3) best-scal))
        (format t "  SIMD Color:   ~6,3F ms (~6,1F Mpix/s)~%" best-simd (/ (* w h 1e-3) best-simd))
        (format t "  Delta:        ~6,3F ms (Speedup: ~,2Fx)~%~%" (- best-scal best-simd) (/ best-scal best-simd))))))

(bench-color-direct 512 512 500)
(bench-color-direct 1024 1024 200)

(sb-ext:exit :code 0)
