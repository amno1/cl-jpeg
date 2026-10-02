(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(sb-alien:load-shared-object "/tmp/librdtsc.so")
(sb-alien:define-alien-routine ("get_cycles" get-cycles) (sb-alien:unsigned 64))

(defun calibrate-tsc ()
  (let* ((c0 (get-cycles))
         (t0 (get-internal-real-time)))
    (sleep 0.1)
    (let* ((c1 (get-cycles))
           (t1 (get-internal-real-time))
           (dt (/ (- t1 t0) (float internal-time-units-per-second)))
           (dc (- c1 c0)))
      (/ dt dc))))

(defun copy-block-2d (src)
  (let ((dst (make-array 8 :element-type 'sint16-array
                           :initial-contents (loop repeat 8 collect (make-array 8 :element-type 'sint16 :initial-element 0)))))
    (dotimes (r 8)
      (let ((row (aref dst r)))
        (dotimes (c 8)
          (setf (aref row c) (s16ref src c r)))))
    dst))

(defun copy-block-into (src dst)
  (dotimes (r 8)
    (let ((srow (aref src r))
          (drow (aref dst r)))
      (dotimes (c 8)
        (setf (aref drow c) (aref srow c))))))

(defun block-sparsity-class (block)
  "Classifies an 8x8 block into :pure-dc, :sparse-ac, or :dense-ac."
  (declare (type sint16-2d-array block)
           (optimize (speed 3) (safety 0)))
  (let ((all-ac-zero t)
        (has-zero-col nil))
    ;; Columns 1..7 (column 0 also has DC, but scalar checks columns 1..7 for zero)
    (loop for col from 0 to 7 do
      (let ((col-ac-zero t))
        (loop for row from 0 to 7 do
          (unless (and (= col 0) (= row 0))
            (unless (zerop (the fixnum (s16ref block col row)))
              (setf col-ac-zero nil
                    all-ac-zero nil))))
        (when col-ac-zero
          (setf has-zero-col t))))
    (cond
      (all-ac-zero :pure-dc)
      (has-zero-col :sparse-ac)
      (t :dense-ac))))

(defun analyze-image-idct (file-path)
  (format t "~%================================================================================~%")
  (format t "  REAL PHOTO IDCT SPARSITY & PER-BLOCK BENCHMARK: ~A~%" file-path)
  (format t "================================================================================~%")
  
  (let* ((sec-per-cycle (calibrate-tsc))
         (ns-per-cycle (* sec-per-cycle 1e9))
         (ws (2d-sint16-array
              '(0 0 0 0 0 0 0 0) '(0 0 0 0 0 0 0 0) '(0 0 0 0 0 0 0 0) '(0 0 0 0 0 0 0 0)
              '(0 0 0 0 0 0 0 0) '(0 0 0 0 0 0 0 0) '(0 0 0 0 0 0 0 0) '(0 0 0 0 0 0 0 0)))
         (blocks nil)
         (q-tabs nil))
    
    ;; Hook into *inverse-llm-dct-impl* to collect all blocks
    (let ((collected-blocks nil)
          (collected-qtabs nil))
      (let ((jpeg:*use-simd* :on)
            (jpeg::*inverse-llm-dct-impl*
              (lambda (b q w)
                (let ((copy (copy-block-2d b)))
                  (push copy collected-blocks)
                  (push q collected-qtabs))
                (inverse-llm-dct-scalar b q w))))
        (decode-image file-path))
      (setf blocks (nreverse collected-blocks)
            q-tabs (nreverse collected-qtabs)))
    
    (let ((total-blocks (length blocks))
          (dc-count 0)
          (sparse-count 0)
          (dense-count 0)
          (dc-scalar-cyc 0)
          (dc-simd-cyc 0)
          (sparse-scalar-cyc 0)
          (sparse-simd-cyc 0)
          (dense-scalar-cyc 0)
          (dense-simd-cyc 0)
          (n-reps 20))
      
      (dolist (b blocks)
        (ecase (block-sparsity-class b)
          (:pure-dc (incf dc-count))
          (:sparse-ac (incf sparse-count))
          (:dense-ac (incf dense-count))))
      
      (format t "  Total 8x8 Blocks: ~:D~%" total-blocks)
      (format t "  - Pure DC Blocks:  ~6:D (~5,1F%)  [All AC = 0]~%"
              dc-count (* 100.0 (/ dc-count (float total-blocks))))
      (format t "  - Sparse AC Blocks: ~6:D (~5,1F%)  [>= 1 zero col shortcut]~%"
              sparse-count (* 100.0 (/ sparse-count (float total-blocks))))
      (format t "  - Dense AC Blocks:  ~6:D (~5,1F%)  [All 8 cols have AC]~%~%"
              dense-count (* 100.0 (/ dense-count (float total-blocks))))
      
      (let ((work-block (copy-block-2d (first blocks))))
        (loop for b in blocks
              for q in q-tabs do
          (let ((cls (block-sparsity-class b)))
            ;; Measure scalar
            (let ((c0 (get-cycles)))
              (dotimes (r n-reps)
                (copy-block-into b work-block)
                (inverse-llm-dct-scalar work-block q ws))
              (let ((cyc (/ (float (- (get-cycles) c0)) n-reps)))
                (ecase cls
                  (:pure-dc (incf dc-scalar-cyc cyc))
                  (:sparse-ac (incf sparse-scalar-cyc cyc))
                  (:dense-ac (incf dense-scalar-cyc cyc)))))
            ;; Measure SIMD
            (let ((c0 (get-cycles)))
              (dotimes (r n-reps)
                (copy-block-into b work-block)
                (inverse-llm-dct-simd work-block q ws))
              (let ((cyc (/ (float (- (get-cycles) c0)) n-reps)))
                (ecase cls
                  (:pure-dc (incf dc-simd-cyc cyc))
                  (:sparse-ac (incf sparse-simd-cyc cyc))
                  (:dense-ac (incf dense-simd-cyc cyc))))))))
      
      (format t "  CATEGORY   | BLOCKS | SCALAR (ns/blk) | SIMD (ns/blk) | KERNEL SPEEDUP | TOTAL SCALAR | TOTAL SIMD~%")
      (format t "  -----------+--------+-----------------+---------------+----------------+--------------+-----------~%")
      (when (> dc-count 0)
        (let* ((sc-ns (* (/ dc-scalar-cyc dc-count) ns-per-cycle))
               (sm-ns (* (/ dc-simd-cyc dc-count) ns-per-cycle))
               (ratio (/ sc-ns sm-ns))
               (tot-sc-ms (* sc-ns dc-count 1e-6))
               (tot-sm-ms (* sm-ns dc-count 1e-6)))
          (format t "  Pure DC    | ~6:D |    ~7,1F ns    |   ~7,1F ns    |     ~5,2Fx      |   ~6,2F ms   |  ~6,2F ms~%"
                  dc-count sc-ns sm-ns ratio tot-sc-ms tot-sm-ms)))
      (when (> sparse-count 0)
        (let* ((sc-ns (* (/ sparse-scalar-cyc sparse-count) ns-per-cycle))
               (sm-ns (* (/ sparse-simd-cyc sparse-count) ns-per-cycle))
               (ratio (/ sc-ns sm-ns))
               (tot-sc-ms (* sc-ns sparse-count 1e-6))
               (tot-sm-ms (* sm-ns sparse-count 1e-6)))
          (format t "  Sparse AC  | ~6:D |    ~7,1F ns    |   ~7,1F ns    |     ~5,2Fx      |   ~6,2F ms   |  ~6,2F ms~%"
                  sparse-count sc-ns sm-ns ratio tot-sc-ms tot-sm-ms)))
      (when (> dense-count 0)
        (let* ((sc-ns (* (/ dense-scalar-cyc dense-count) ns-per-cycle))
               (sm-ns (* (/ dense-simd-cyc dense-count) ns-per-cycle))
               (ratio (/ sc-ns sm-ns))
               (tot-sc-ms (* sc-ns dense-count 1e-6))
               (tot-sm-ms (* sm-ns dense-count 1e-6)))
          (format t "  Dense AC   | ~6:D |    ~7,1F ns    |   ~7,1F ns    |     ~5,2Fx      |   ~6,2F ms   |  ~6,2F ms~%"
                  dense-count sc-ns sm-ns ratio tot-sc-ms tot-sm-ms)))
      
      (let* ((total-scalar-ms (* (+ dc-scalar-cyc sparse-scalar-cyc dense-scalar-cyc) ns-per-cycle 1e-6))
             (total-simd-ms   (* (+ dc-simd-cyc sparse-simd-cyc dense-simd-cyc) ns-per-cycle 1e-6))
             (overall-ratio   (/ total-scalar-ms total-simd-ms)))
        (format t "  -----------+--------+-----------------+---------------+----------------+--------------+-----------~%")
        (format t "  OVERALL    | ~6:D |    ~7,1F ns    |   ~7,1F ns    |     ~5,2Fx      |   ~6,2F ms   |  ~6,2F ms~%"
                total-blocks
                (* (/ total-scalar-ms total-blocks) 1e6)
                (* (/ total-simd-ms total-blocks) 1e6)
                overall-ratio
                total-scalar-ms
                total-simd-ms))
      (format t "================================================================================~%"))))

(analyze-image-idct "/tmp/photo1_512x512.jpg")
(analyze-image-idct "/tmp/photo3_1024x1024.jpg")
(analyze-image-idct "/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg")

(sb-ext:exit :code 0)
