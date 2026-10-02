(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun bench-runs (reps)
  (let ((path "/tmp/photo1_512x512.jpg"))
    (format t "~%=== BENCHMARKING photo1_512x512.jpg (~D iterations) ===~%" reps)
    (dotimes (i 5) (decode-image path))

    ;; 1. Default system state (simd on, idct simd, color simd)
    (let ((t0 (get-internal-real-time)))
      (dotimes (i reps) (decode-image path))
      (format t "1. Default system state (SIMD IDCT + SIMD Color):  ~6,2F ms~%"
              (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0)))

    ;; 2. *use-simd* = :off
    (let ((*use-simd* :off))
      (let ((t0 (get-internal-real-time)))
        (dotimes (i reps) (decode-image path))
        (format t "2. Pure Scalar (*use-simd* :off):                  ~6,2F ms~%"
                (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0))))

    ;; 3. *use-simd* = :on, idct = scalar, color = scalar
    (let ((*use-simd* :on)
          (*inverse-llm-dct-impl* #'inverse-llm-dct-scalar)
          (*inverse-colorspace-convert-impl* #'inverse-colorspace-convert))
      (let ((t0 (get-internal-real-time)))
        (dotimes (i reps) (decode-image path))
        (format t "3. :on with Scalar IDCT + Scalar Color:             ~6,2F ms~%"
                (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0))))

    ;; 4. *use-simd* = :on, idct = simd, color = scalar
    (let ((*use-simd* :on)
          (*inverse-llm-dct-impl* #'inverse-llm-dct-simd)
          (*inverse-colorspace-convert-impl* #'inverse-colorspace-convert))
      (let ((t0 (get-internal-real-time)))
        (dotimes (i reps) (decode-image path))
        (format t "4. :on with SIMD IDCT + Scalar Color:               ~6,2F ms~%"
                (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0))))

    ;; 5. *use-simd* = :on, idct = stubbed, color = scalar
    (let ((*use-simd* :on)
          (*inverse-llm-dct-impl* (lambda (b q w) (declare (ignore b q w)) (values)))
          (*inverse-colorspace-convert-impl* #'inverse-colorspace-convert))
      (let ((t0 (get-internal-real-time)))
        (dotimes (i reps) (decode-image path))
        (format t "5. :on with STUBBED IDCT + Scalar Color:            ~6,2F ms~%"
                (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0))))

    ;; 6. *use-simd* = :on, idct = stubbed, color = simd
    (let ((*use-simd* :on)
          (*inverse-llm-dct-impl* (lambda (b q w) (declare (ignore b q w)) (values)))
          (*inverse-colorspace-convert-impl* #'inverse-colorspace-convert-simd-accelerated))
      (let ((t0 (get-internal-real-time)))
        (dotimes (i reps) (decode-image path))
        (format t "6. :on with STUBBED IDCT + SIMD Color:              ~6,2F ms~%"
                (* (/ (- (get-internal-real-time) t0) (* (float reps) internal-time-units-per-second)) 1000.0))))))

(bench-runs 100)
(sb-ext:exit :code 0)
