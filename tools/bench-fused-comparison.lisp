(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun bench-one-config (path reps &key planar descriptor)
  "Run REPS decodes of PATH, returning (values ms-per-image bytes-consed-per-image)."
  (let* ((*upsample-impl* (if planar #'planar-upsample nil))
         (*use-simd* :on)
         (desc (if descriptor (or descriptor (make-descriptor)) nil)))
    ;; Warmup
    (dotimes (i 5)
      (decode-image path :descriptor desc))
    ;; Measure
    (sb-ext:gc :full t)
    (let ((c0 (sb-ext:get-bytes-consed))
          (t0 (get-internal-real-time)))
      (dotimes (i reps)
        (decode-image path :descriptor desc))
      (let* ((t1 (get-internal-real-time))
             (c1 (sb-ext:get-bytes-consed))
             (dt (/ (- t1 t0) (float internal-time-units-per-second)))
             (ms-per-img (* (/ dt reps) 1000.0))
             (bytes-per-img (/ (- c1 c0) (float reps))))
        (values ms-per-img bytes-per-img)))))

(defun run-fused-benchmark ()
  (format t "====================================================================================================~%")
  (format t "  CL-JPEG BENCHMARK: INTERLEAVED SIMD VS FUSED PLANAR SIMD (Zen 5 AVX-512)~%")
  (format t "====================================================================================================~%")
  (format t "  Image                 | Mode        | Single-Shot (ms) | Consed (MB) | Reused (ms) | Consed (MB) ~%")
  (format t "  ----------------------+-------------+------------------+-------------+-------------+-------------~%")

  (dolist (item (list (list "/tmp/photo1_512x512.jpg" "photo1 512x512" 60)
                      (list "/tmp/photo3_1024x1024.jpg" "photo3 1024x1024" 30)
                      (list "/tmp/cl-jpeg-bench-512x512.jpg" "gradient 512x512" 60)
                      (list "/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg" "testorig 227x149" 100)))
    (let ((path (first item))
          (name (second item))
          (reps (third item)))
      (when (probe-file path)
        ;; Interleaved
        (multiple-value-bind (int-single-ms int-single-bytes)
            (bench-one-config path reps :planar nil :descriptor nil)
          (multiple-value-bind (int-reuse-ms int-reuse-bytes)
              (bench-one-config path reps :planar nil :descriptor (make-descriptor))
            ;; Fused Planar
            (multiple-value-bind (fus-single-ms fus-single-bytes)
                (bench-one-config path reps :planar t :descriptor nil)
              (multiple-value-bind (fus-reuse-ms fus-reuse-bytes)
                  (bench-one-config path reps :planar t :descriptor (make-descriptor))
                (format t "  ~21A | Interleaved | ~14,2F ms | ~9,2F MB | ~9,2F ms | ~9,2F MB~%"
                        name int-single-ms (/ int-single-bytes 1048576.0)
                        int-reuse-ms (/ int-reuse-bytes 1048576.0))
                (format t "  ~21A | Fused       | ~14,2F ms | ~9,2F MB | ~9,2F ms | ~9,2F MB~%"
                        "" fus-single-ms (/ fus-single-bytes 1048576.0)
                        fus-reuse-ms (/ fus-reuse-bytes 1048576.0))
                (format t "  ~21A | Speedup/Δ   | ~13,2Fx   | ~+8,2F MB | ~8,2Fx   | ~+8,2F MB~%"
                        "" (/ int-single-ms fus-single-ms)
                        (/ (- fus-single-bytes int-single-bytes) 1048576.0)
                        (/ int-reuse-ms fus-reuse-ms)
                        (/ (- fus-reuse-bytes int-reuse-bytes) 1048576.0))
                (format t "  ----------------------+-------------+------------------+-------------+-------------+-------------~%")))))))))

(run-fused-benchmark)
