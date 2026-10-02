(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun run-timed-batch (path reps &key (use-simd :on) (idct :scalar) (color :scalar) (upsample :default))
  ;; Configure hooks
  (let ((jpeg:*use-simd* use-simd)
        (jpeg::*inverse-llm-dct-impl*
          (ecase idct
            (:scalar #'jpeg::inverse-llm-dct-scalar)
            (:simd   #'jpeg::inverse-llm-dct-simd)
            (:stub   (lambda (b q w) (declare (ignore b q w)) (values)))))
        (jpeg::*inverse-colorspace-convert-impl*
          (ecase color
            (:scalar #'jpeg::inverse-colorspace-convert)
            (:simd   #'jpeg::inverse-colorspace-convert-simd-accelerated)))
        (jpeg::*upsample-impl*
          (ecase upsample
            (:interleaved nil)
            (:planar      #'jpeg::planar-upsample)
            (:default     (if (eql use-simd :off) nil #'jpeg::planar-upsample)))))
    ;; Warmup
    (dotimes (i 5) (decode-image path))
    ;; Best of 5 batches
    (let ((best-ms 1e9))
      (dotimes (b 5)
        (sb-ext:gc :full t)
        (let ((t0 (get-internal-real-time)))
          (dotimes (r reps)
            (decode-image path))
          (let* ((t1 (get-internal-real-time))
                 (ms (* (/ (- t1 t0) (float internal-time-units-per-second)) (/ 1000.0 reps))))
            (when (< ms best-ms)
              (setf best-ms ms)))))
      best-ms)))

(defun run-ablation-suite ()
  (format t "~%========================================================================================================================~%")
  (format t "  UNIFIED DECODER ABLATION & ROADMAP PERFORMANCE (AMD Zen 5, AVX2/AVX-512, 100 reps)~%")
  (format t "========================================================================================================================~%")
  (format t "  Image         | All-Scalar | Scal IDCT | SIMD IDCT | Stub IDCT | In-Situ | Interleaved | Planar SIMD | Total Speedup~%")
  (format t "                | (:off)     | +Scal Col | +Scal Col | +Scal Col | IDCT    | SIMD Decode | (Full SIMD) | vs All-Scalar~%")
  (format t "  --------------+------------+-----------+-----------+-----------+---------+-------------+-------------+--------------~%")
  
  (dolist (item (list (list "/tmp/photo1_512x512.jpg" "photo1 512x512" 100)
                      (list "/tmp/photo3_1024x1024.jpg" "photo3 1024x1024" 30)
                      (list "/tmp/cl-jpeg-bench-512x512.jpg" "gradient 512x512" 100)))
    (let* ((path (first item))
           (label (second item))
           (reps (third item)))
      (when (probe-file path)
        (let* ((t-off       (run-timed-batch path reps :use-simd :off :idct :scalar :color :scalar :upsample :interleaved))
               (t-scal-scal (run-timed-batch path reps :use-simd :on  :idct :scalar :color :scalar :upsample :interleaved))
               (t-simd-scal (run-timed-batch path reps :use-simd :on  :idct :simd   :color :scalar :upsample :interleaved))
               (t-stub-scal (run-timed-batch path reps :use-simd :on  :idct :stub   :color :scalar :upsample :interleaved))
               (t-int-simd  (run-timed-batch path reps :use-simd :on  :idct :simd   :color :simd   :upsample :interleaved))
               (t-pln-simd  (run-timed-batch path reps :use-simd :on  :idct :simd   :color :simd   :upsample :planar))
               (net-scal    (- t-scal-scal t-stub-scal))
               (net-simd    (- t-simd-scal t-stub-scal))
               (in-situ     (if (> net-simd 0.05) (/ net-scal net-simd) 0.0))
               (speedup     (/ t-off t-pln-simd)))
          (format t "  ~13A |  ~6,2F ms  | ~6,2F ms | ~6,2F ms | ~6,2F ms |  ~5,2Fx  |   ~6,2F ms  |   ~6,2F ms  |     ~5,2Fx~%"
                  label t-off t-scal-scal t-simd-scal t-stub-scal in-situ t-int-simd t-pln-simd speedup)))))
  (format t "========================================================================================================================~%"))

(run-ablation-suite)
(sb-ext:exit :code 0)
