(in-package :cl-user)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (load "~/quicklisp/setup.lisp")
  (asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
  (asdf:load-system :cl-jpeg)
  (require :sb-simd))

(in-package :jpeg)

(defun measure-decode (path n-reps &key simd idct-simd color-simd)
  (let ((jpeg:*use-simd* (if simd :on :off))
        (jpeg::*inverse-colorspace-convert-impl* (if color-simd #'jpeg::inverse-colorspace-convert-simd-accelerated nil))
        (jpeg::*inverse-llm-dct-impl* (if idct-simd #'jpeg::inverse-llm-dct-simd nil)))
    ;; Warmup
    (dotimes (i 5) (jpeg:decode-image path))
    ;; Measure best of 5 batches
    (let ((best 1e9))
      (dotimes (b 5)
        (sb-ext:gc :full t)
        (let ((t0 (get-internal-real-time)))
          (dotimes (i n-reps)
            (jpeg:decode-image path))
          (let ((elapsed (* (/ (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second)) n-reps) 1000.0)))
            (when (< elapsed best) (setf best elapsed)))))
      best)))


(defun benchmark-decoder (test-files)
  (format t "========================================================================================~%")
  (format t "  CL-JPEG DECODER ROADMAP PERFORMANCE SUMMARY (Zen 5 AVX2/AVX-512)~%")
  (format t "========================================================================================~%")
  (format t "  Image                        | Scalar IDCT | SIMD IDCT   | Speedup (IDCT) | Full SIMD~%")
  (format t "  -----------------------------+-------------+-------------+----------------+-----------~%")
  (dolist (f test-files)
    (when (probe-file f)
      (let* ((reps (cond ((search "1024" f) 15) ((search "2048" f) 5) (t 40)))
             (ms-scalar-idct (measure-decode f reps :simd :on :idct-simd nil :color-simd nil))
             (ms-simd-idct   (measure-decode f reps :simd :on :idct-simd t   :color-simd nil))
             (ms-full-simd   (measure-decode f reps :simd :on :idct-simd t   :color-simd t))
             (ratio-idct (/ ms-scalar-idct ms-simd-idct))
             (ratio-full (/ ms-scalar-idct ms-full-simd)))
        (format t "  ~28A | ~8,2F ms | ~8,2F ms | ~12,2Fx | ~7,2F ms (~,2Fx)~%"
                (file-namestring f)
                ms-scalar-idct
                ms-simd-idct
                ratio-idct
                ms-full-simd
                ratio-full))))
  (format t "========================================================================================~%"))

;; Encode test images if needed
(defun ensure-test-jpegs ()
  (let ((raw1 "/home/arthur/repos/cl/cl-jpeg-avx512/tools/test-images/photo1_512x512.raw")
        (raw3 "/home/arthur/repos/cl/cl-jpeg-avx512/tools/test-images/photo3_1024x1024.raw"))
    (when (probe-file raw1)
      (let ((buf (make-array (* 512 512 3) :element-type 'uint8)))
        (with-open-file (s raw1 :element-type 'uint8)
          (read-sequence buf s))
        (jpeg:encode-image "/tmp/photo1_512x512.jpg" buf 3 512 512 :q-factor 25 :chroma-subsampling :box)))
    (when (probe-file raw3)
      (let ((buf (make-array (* 1024 1024 3) :element-type 'uint8)))
        (with-open-file (s raw3 :element-type 'uint8)
          (read-sequence buf s))
        (jpeg:encode-image "/tmp/photo3_1024x1024.jpg" buf 3 1024 1024 :q-factor 25 :chroma-subsampling :box)))))

(ensure-test-jpegs)

(benchmark-decoder (list "/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg"
                         "/home/arthur/repos/libjpeg-turbo/testimages/testorig_420_Q95.jpg"
                         "/tmp/photo1_512x512.jpg"
                         "/tmp/photo3_1024x1024.jpg"))
