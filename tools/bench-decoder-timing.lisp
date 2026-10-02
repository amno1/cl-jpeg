;;; bench-decoder-timing.lisp --- Meticulous Manual Timer Profiling for CL-JPEG Decoder
;;;
;;; Direct hardware TSC cycle instrumentation and ablation breakdown of:
;;;   1. Huffman entropy decoding (decode-block: decode-dc + decode-ac + bit reader)
;;;   2. Inverse zigzag (izigzag)
;;;   3. DC predictor accumulation
;;;   4. Inverse LLM-DCT & dequantization (inverse-llm-dct)
;;;   5. Block upsampling & buffer write (upsample)
;;;   6. Inverse colorspace conversion (YCbCr -> RGB, scalar and SIMD)
;;;   7. Header, markers, and stream I/O overhead

(require :asdf)
(push #p"/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(sb-alien:load-shared-object "/tmp/librdtsc.so")
(sb-alien:define-alien-routine ("get_cycles" get-cycles) (sb-alien:unsigned 64))

(defvar *timer-active* nil)
(defvar *stat-huffman* 0)
(defvar *stat-izigzag* 0)
(defvar *stat-dcpred* 0)
(defvar *stat-idct* 0)
(defvar *stat-upsample* 0)
(defvar *stat-blocks* 0)
(defvar *stat-color* 0)

(declaim (type (unsigned-byte 64)
               *stat-huffman* *stat-izigzag* *stat-dcpred*
               *stat-idct* *stat-upsample* *stat-blocks* *stat-color*))

(defun reset-decoder-stats ()
  (setf *stat-huffman* 0
        *stat-izigzag* 0
        *stat-dcpred* 0
        *stat-idct* 0
        *stat-upsample* 0
        *stat-blocks* 0
        *stat-color* 0))

;;; Instrumented decode-chunk
(defun decode-chunk-instrumented (image scan zzbuf)
  "Reads and decodes either a whole scan (if no restarts) or restart interval with per-stage timers."
  (let* ((nextbit (make-nextbit 0 0))
         (ncomp (scan-ncomp scan))
         (nwidth (mul (descriptor-width image) (descriptor-ncomp image)))
         (dend (mul (descriptor-height image) nwidth))
         (fr (make-array ncomp :initial-contents
                         (loop
                           for i fixnum from 0 below ncomp
                           for cid fixnum = (first (aref (scan-cdesc scan) i))
                           for pos fixnum = (position cid (descriptor-cid image))
                           collecting (list (aref (descriptor-H image) pos)
                                            (aref (descriptor-V image) pos)))))
         (Hmax (loop for entry across fr maximize (the fixnum (first entry))))
         (Vmax (loop for entry across fr maximize (the fixnum (second entry))))
         (x-growth (ash Hmax 3))
         (y-growth (ash Vmax 3))
         (freqs (make-array ncomp :initial-contents
                            (loop for i fixnum from 0 below ncomp
                                  for cid fixnum = (first (aref (scan-cdesc scan) i))
                                  for pos fixnum = (position cid (descriptor-cid image))
                                  collecting (list (aref (descriptor-iH image) pos)
                                                   (aref (descriptor-iV image) pos)))))
         (preds (make-array ncomp :initial-element 0 :element-type 'sint16))
         (tables (make-array ncomp
                     :initial-contents
                     (loop for i fixnum from 0 below ncomp
                           for ta fixnum = (logand (second (aref (scan-cdesc scan) i)) 15)
                           for td fixnum = (ash (the fixnum (second (aref (scan-cdesc scan) i))) -4)
                           collecting (vector (aref (descriptor-huff-ac image) ta)
                                              (aref (descriptor-huff-dc image) td))))))
    (declare #.*optimize*
             (type fixnum ncomp Hmax Vmax x-growth y-growth nwidth)
             (type (simple-array t (*)) freqs fr tables)
             (type (simple-array sint16 (*)) preds)
             (dynamic-extent fr freqs))
    (catch 'marker
      (loop
        do (loop for comp fixnum from 0 below ncomp
                 for pos fixnum =
                    (position (the fixnum (first (aref (scan-cdesc scan) comp)))
                              (descriptor-cid image))
                 for q-tab = (aref (descriptor-qtables image) (aref (descriptor-qdest image) comp))
                 for H fixnum = (first (aref freqs comp))
                 for V fixnum = (second (aref freqs comp))
                 for nw fixnum = (mul nwidth V)
                 for nx fixnum = (mul (descriptor-ncomp image) H)
                 for blocks-y fixnum = (second (aref fr comp))
                 for blocks-x fixnum = (first (aref fr comp)) do
                    (loop for y fixnum from 0 below blocks-y
                          for y-pos fixnum from (mul (ash y 3) V) by (ash V 3) do
                            (loop for x fixnum from 0 below blocks-x
                                  for x-pos fixnum from (mul (ash x 3) H) by (ash H 3) do
                                    (if (not *timer-active*)
                                        ;; Fast uninstrumented path
                                        (let ((decoded-block
                                                (izigzag (decode-block (descriptor-zz image)
                                                                       (aref tables comp) nextbit image) zzbuf)))
                                          (incf (s16ref decoded-block 0 0) (aref preds comp))
                                          (setf (aref preds comp) (s16ref decoded-block 0 0))
                                          (when (and (< (plus x-pos (scan-x scan)) (descriptor-width image))
                                                     (< (plus y-pos (scan-y scan)) (descriptor-height image)))
                                            (inverse-llm-dct decoded-block q-tab (descriptor-ws image))
                                            (upsample image scan decoded-block x-pos y-pos
                                                      H V pos nwidth nw nx dend)))
                                        ;; Precise instrumented path
                                        (let* ((c0 (get-cycles))
                                               (zz (decode-block (descriptor-zz image)
                                                                 (aref tables comp) nextbit image))
                                               (c1 (get-cycles))
                                               (decoded-block (izigzag zz zzbuf))
                                               (c2 (get-cycles)))
                                          (incf (s16ref decoded-block 0 0) (aref preds comp))
                                          (setf (aref preds comp) (s16ref decoded-block 0 0))
                                          (let ((c3 (get-cycles)))
                                            (when (and (< (plus x-pos (scan-x scan)) (descriptor-width image))
                                                       (< (plus y-pos (scan-y scan)) (descriptor-height image)))
                                              (inverse-llm-dct decoded-block q-tab (descriptor-ws image))
                                              (let ((c4 (get-cycles)))
                                                (upsample image scan decoded-block x-pos y-pos
                                                          H V pos nwidth nw nx dend)
                                                (let ((c5 (get-cycles)))
                                                  (incf *stat-huffman* (- c1 c0))
                                                  (incf *stat-izigzag* (- c2 c1))
                                                  (incf *stat-dcpred* (- c3 c2))
                                                  (incf *stat-idct* (- c4 c3))
                                                  (incf *stat-upsample* (- c5 c4))
                                                  (incf *stat-blocks*))))))))))
           (incf (scan-x scan) x-growth)
           (when (<= (descriptor-width image) (scan-x scan))
             (incf (scan-y scan) y-growth)
             (setf (scan-x scan) 0))))))

;; Shadow original decode-chunk with instrumented version
(setf (fdefinition 'decode-chunk) #'decode-chunk-instrumented)

;; Instrumented colorspace conversion wrapper
(let ((orig-colorspace (fdefinition 'inverse-colorspace-convert-dispatch)))
  (defun inverse-colorspace-convert-instrumented (descriptor)
    (if *timer-active*
        (let ((c0 (get-cycles)))
          (funcall orig-colorspace descriptor)
          (let ((c1 (get-cycles)))
            (incf *stat-color* (- c1 c0))))
        (funcall orig-colorspace descriptor))))

(setf (fdefinition 'inverse-colorspace-convert-dispatch) #'inverse-colorspace-convert-instrumented)

(in-package :cl-user)

(defun run-decoder-profile (file-path &key (reps 10) (warmup 3))
  (format t "~%================================================================~%")
  (format t " DECODER PROFILE FOR: ~A~%" file-path)
  (format t "================================================================~%")
  
  ;; 1. Measure TSC to calibrate invariant frequency
  (let* ((c_start (jpeg::get-cycles))
         (t_start (get-internal-real-time)))
    (sleep 0.1)
    (let* ((c_end (jpeg::get-cycles))
           (t_end (get-internal-real-time))
           (dt (/ (- t_end t_start) (float internal-time-units-per-second)))
           (dc (- c_end c_start))
           (tsc-freq (/ dc dt))
           (ns-per-cycle (* (/ 1.0 tsc-freq) 1e9)))
      
      ;; 2. Warm up CPU & caches with uninstrumented runs
      (setf jpeg::*timer-active* nil)
      (sb-ext:gc :full t)
      (dotimes (i warmup)
        (jpeg:decode-image file-path))
      
      ;; 3. Baseline uninstrumented total decode timing (Zero overhead)
      (let ((uninst-times nil))
        (dotimes (i reps)
          (let ((c0 (jpeg::get-cycles)))
            (jpeg:decode-image file-path)
            (let ((c1 (jpeg::get-cycles)))
              (push (* (- c1 c0) ns-per-cycle 1e-6) uninst-times))))
        (setf uninst-times (sort uninst-times #'<))
        (let ((median-uninst (nth (floor reps 2) uninst-times))
              (mean-uninst (/ (reduce #'+ uninst-times) (length uninst-times))))
          (format t "~%--- 1. Baseline Uninstrumented Total Time ---~%")
          (format t "  Repetitions: ~D (Warmup: ~D)~%" reps warmup)
          (format t "  Median Total: ~8,3F ms~%" median-uninst)
          (format t "  Mean Total:   ~8,3F ms~%" mean-uninst)
          
          ;; 4. Direct In-Loop Manual Timer Breakdown
          (jpeg::reset-decoder-stats)
          (setf jpeg::*timer-active* t)
          (sb-ext:gc :full t)
          (let ((total-inst-cycles 0))
            (dotimes (i reps)
              (let ((c0 (jpeg::get-cycles)))
                (jpeg:decode-image file-path)
                (let ((c1 (jpeg::get-cycles)))
                  (incf total-inst-cycles (- c1 c0)))))
            
            (let* ((total-blocks (/ jpeg::*stat-blocks* reps))
                   (huff-cyc (/ (float jpeg::*stat-huffman*) reps))
                   (izig-cyc (/ (float jpeg::*stat-izigzag*) reps))
                   (pred-cyc (/ (float jpeg::*stat-dcpred*) reps))
                   (idct-cyc (/ (float jpeg::*stat-idct*) reps))
                   (upsm-cyc (/ (float jpeg::*stat-upsample*) reps))
                   (colr-cyc (/ (float jpeg::*stat-color*) reps))
                   (tot-cyc  (/ (float total-inst-cycles) reps))
                   
                   ;; Milliseconds per iteration
                   (huff-ms (* huff-cyc ns-per-cycle 1e-6))
                   (izig-ms (* izig-cyc ns-per-cycle 1e-6))
                   (pred-ms (* pred-cyc ns-per-cycle 1e-6))
                   (idct-ms (* idct-cyc ns-per-cycle 1e-6))
                   (upsm-ms (* upsm-cyc ns-per-cycle 1e-6))
                   (colr-ms (* colr-cyc ns-per-cycle 1e-6))
                   (tot-ms  (* tot-cyc  ns-per-cycle 1e-6))
                   
                   (sum-accounted (+ huff-ms izig-ms pred-ms idct-ms upsm-ms colr-ms))
                   (overhead-ms (- tot-ms sum-accounted))
                   
                   ;; Percentages relative to uninstrumented baseline median
                   (pct-huff (* (/ huff-ms median-uninst) 100.0))
                   (pct-izig (* (/ izig-ms median-uninst) 100.0))
                   (pct-pred (* (/ pred-ms median-uninst) 100.0))
                   (pct-idct (* (/ idct-ms median-uninst) 100.0))
                   (pct-upsm (* (/ upsm-ms median-uninst) 100.0))
                   (pct-colr (* (/ colr-ms median-uninst) 100.0))
                   (pct-over (* (/ overhead-ms median-uninst) 100.0))
                   
                   ;; Nanoseconds per block
                   (ns-per-block-huff (* (/ huff-ms total-blocks) 1e6))
                   (ns-per-block-izig (* (/ izig-ms total-blocks) 1e6))
                   (ns-per-block-idct (* (/ idct-ms total-blocks) 1e6))
                   (ns-per-block-upsm (* (/ upsm-ms total-blocks) 1e6)))
              
              (format t "~%--- 2. Manual Timer Stage Breakdown (Averaged over ~D runs) ---~%" reps)
              (format t "  Total 8x8 Blocks: ~:D~%" (round total-blocks))
              (format t "  Timer Overhead Inflation: ~+5,2F% (Inst: ~,3F ms vs Uninst: ~,3F ms)~%~%"
                      (* 100.0 (/ (- tot-ms median-uninst) median-uninst))
                      tot-ms median-uninst)
              
              (format t "  STAGE                   |   TIME (ms)  |  %% OF BASELINE |   ns/BLOCK~%")
              (format t "  ------------------------+--------------+-----------------+------------~%")
              (format t "  Huffman Entropy Decode  |  ~8,3F ms |      ~5,1F %%    |  ~7,1F ns~%"
                      huff-ms pct-huff ns-per-block-huff)
              (format t "  Inverse Zigzag (izigzag)|  ~8,3F ms |      ~5,1F %%    |  ~7,1F ns~%"
                      izig-ms pct-izig ns-per-block-izig)
              (format t "  DC Predictor Update     |  ~8,3F ms |      ~5,1F %%    |  ~7,1F ns~%"
                      pred-ms pct-pred (* (/ pred-ms total-blocks) 1e6))
              (format t "  Inverse LLM-DCT + Dequ  |  ~8,3F ms |      ~5,1F %%    |  ~7,1F ns~%"
                      idct-ms pct-idct ns-per-block-idct)
              (format t "  Upsample to Component   |  ~8,3F ms |      ~5,1F %%    |  ~7,1F ns~%"
                      upsm-ms pct-upsm ns-per-block-upsm)
              (format t "  YCbCr->RGB Colorspace   |  ~8,3F ms |      ~5,1F %%    |    ---    ~%"
                      colr-ms pct-colr)
              (format t "  I/O, Markers & Scan Loop|  ~8,3F ms |      ~5,1F %%    |    ---    ~%"
                      overhead-ms pct-over)
              (format t "  ------------------------+--------------+-----------------+------------~%")
              (format t "  Total Accounted         |  ~8,3F ms |      ~5,1F %%    |  ~7,1F ns~%"
                      tot-ms (* (/ tot-ms median-uninst) 100.0) (* (/ tot-ms total-blocks) 1e6))
              
              ;; 5. Ablation Confirmation (Colorspace OFF)
              (setf jpeg::*timer-active* nil)
              (let ((no-color-times nil))
                (dotimes (i reps)
                  (let ((c0 (jpeg::get-cycles)))
                    (jpeg:decode-image file-path :colorspace-conversion nil)
                    (let ((c1 (jpeg::get-cycles)))
                      (push (* (- c1 c0) ns-per-cycle 1e-6) no-color-times))))
                (setf no-color-times (sort no-color-times #'<))
                (let* ((med-no-color (nth (floor reps 2) no-color-times))
                       (ablation-color-delta (- median-uninst med-no-color)))
                  (format t "~%--- 3. Ablation Verification (Zero-Overhead Color Delta) ---~%")
                  (format t "  Decode without colorspace: ~8,3F ms~%" med-no-color)
                  (format t "  Color conversion delta:    ~8,3F ms (Manual timer measured: ~,3F ms)~%"
                          ablation-color-delta colr-ms))))))))))

(defun run-all ()
  (format t "################################################################~%")
  (format t "  CL-JPEG DECODER MANUAL TIMER INVESTIGATION & AUDIT~%")
  (format t "  Architecture: Zen 5 | SBCL with SB-SIMD-512~%")
  (format t "################################################################~%")
  
  (run-decoder-profile "/tmp/photo1_512x512.jpg" :reps 20 :warmup 5)
  (run-decoder-profile "/tmp/photo3_1024x1024.jpg" :reps 10 :warmup 5)
  (run-decoder-profile "/tmp/grad_2048x2048.jpg" :reps 5 :warmup 3)
  (run-decoder-profile "/tmp/moire_2048x2048.jpg" :reps 5 :warmup 3))

(run-all)
(sb-ext:exit)
