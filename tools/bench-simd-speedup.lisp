;;;; bench-simd-speedup.lisp --- Benchmark Phase 1A AVX-512 Inverse Color Conversion
(require :asdf)
(load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname)))
(push (truename "/home/arthur/repos/cl/cl-jpeg-avx512") asdf:*central-registry*)
(asdf:load-system "cl-jpeg")

(in-package :jpeg)

(defun bench-color-conversion (w h &key (iterations 50))
  (let* ((n (* w h))
         (buf-len (* n 3))
         (buf-scal (make-array buf-len :element-type '(unsigned-byte 8) :initial-element 128))
         (buf-simd (make-array buf-len :element-type '(unsigned-byte 8) :initial-element 128))
         (desc-scal (make-descriptor :width w :height h :buffer buf-scal))
         (desc-simd (make-descriptor :width w :height h :buffer buf-simd)))

    ;; Warmup
    (dotimes (i 5)
      (inverse-colorspace-convert desc-scal)
      (inverse-colorspace-convert-simd-accelerated desc-simd))

    ;; Scalar timing
    (let ((t0 (get-internal-real-time)))
      (dotimes (i iterations)
        (inverse-colorspace-convert desc-scal))
      (let* ((t1 (get-internal-real-time))
             (sec-scal (/ (- t1 t0) (* (float iterations) internal-time-units-per-second)))
             (ms-scal (* sec-scal 1000.0))
             (mpix-scal (/ (* n 1e-6) sec-scal)))

        ;; SIMD timing
        (let ((t2 (get-internal-real-time)))
          (dotimes (i iterations)
            (inverse-colorspace-convert-simd-accelerated desc-simd))
          (let* ((t3 (get-internal-real-time))
                 (sec-simd (/ (- t3 t2) (* (float iterations) internal-time-units-per-second)))
                 (ms-simd (* sec-simd 1000.0))
                 (mpix-simd (/ (* n 1e-6) sec-simd))
                 (speedup (/ sec-scal sec-simd)))
            (format t "  ~5Dx~5D (~6,2F MP): Scalar: ~6,2F ms (~7,1F MP/s) | SIMD: ~6,3F ms (~7,1F MP/s) | Speedup: ~5,2Fx~%"
                    w h (* n 1e-6) ms-scal mpix-scal ms-simd mpix-simd speedup)
            (values ms-scal ms-simd speedup)))))))

(defun bench-end-to-end-decode (w h &key (iterations 20))
  (let* ((n (* w h))
         (rgb (make-array (* n 3) :element-type '(unsigned-byte 8)))
         (out-path (format nil "/tmp/cl-jpeg-bench-~Dx~D.jpg" w h)))
    ;; Fill with benchmark gradient pattern
    (dotimes (y h)
      (dotimes (x w)
        (let ((idx (* (+ (* y w) x) 3)))
          (setf (aref rgb idx)       (logand (+ x y) 255)
                (aref rgb (+ idx 1)) (logand (* x 2) 255)
                (aref rgb (+ idx 2)) (logand (* y 2) 255)))))
    (encode-image (pathname out-path) rgb 3 h w)

    ;; Warmup
    (let ((*use-simd* :off))
      (dotimes (i 3) (decode-image (pathname out-path))))
    (let ((*use-simd* :auto))
      (dotimes (i 3) (decode-image (pathname out-path))))

    ;; Benchmark Scalar Decode
    (let ((t0 (get-internal-real-time)))
      (dotimes (i iterations)
        (let ((*use-simd* :off))
          (decode-image (pathname out-path))))
      (let* ((t1 (get-internal-real-time))
             (sec-scal (/ (- t1 t0) (* (float iterations) internal-time-units-per-second)))
             (ms-scal (* sec-scal 1000.0))
             (mpix-scal (/ (* n 1e-6) sec-scal)))

        ;; Benchmark SIMD Decode
        (let ((t2 (get-internal-real-time)))
          (dotimes (i iterations)
            (let ((*use-simd* :auto))
              (decode-image (pathname out-path))))
          (let* ((t3 (get-internal-real-time))
                 (sec-simd (/ (- t3 t2) (* (float iterations) internal-time-units-per-second)))
                 (ms-simd (* sec-simd 1000.0))
                 (mpix-simd (/ (* n 1e-6) sec-simd))
                 (speedup (/ sec-scal sec-simd)))
            (format t "  Full Decode ~5Dx~5D: Scalar: ~6,2F ms (~6,1F MP/s) | SIMD: ~6,2F ms (~6,1F MP/s) | Total Decode Speedup: ~5,2Fx~%"
                    w h ms-scal mpix-scal ms-simd mpix-simd speedup)))))))

(defun main ()
  (format t "========================================================================================~%")
  (format t "  CL-JPEG AVX-512 Benchmark: Phase 1A (YCbCr -> RGB Inverse Color Conversion)~%")
  (format t "  Target: AMD Zen 5 (Native 512-bit vector units, AVX512F+BW+VBMI)~%")
  (format t "========================================================================================~%")

  (format t "~%--- Microbenchmark: Isolated Inverse Color Conversion ---~%")
  (bench-color-conversion 512 512 :iterations 100)
  (bench-color-conversion 1024 1024 :iterations 50)
  (bench-color-conversion 2048 2048 :iterations 20)
  (bench-color-conversion 4096 4096 :iterations 5)

  (format t "~%--- Macrobenchmark: Full Pipeline End-to-End Decode (Native Files, Matching Baseline) ---~%")
  (bench-end-to-end-decode 512 512 :iterations 30)
  (bench-end-to-end-decode 1024 1024 :iterations 20)
  (bench-end-to-end-decode 2048 2048 :iterations 10)

  (format t "~%========================================================================================~%"))

(main)
