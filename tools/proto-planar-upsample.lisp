;;; tools/proto-planar-upsample.lisp
;;;
;;; Prototype and benchmark for Planar Chroma Intermediate architecture:
;;; 1. upsample-1x1-planar (contiguous 8-byte rows)
;;; 2. upsample-2x2-planar (vpunpcklbw doubling 8 bytes to 16 bytes across 2 rows)
;;; 3. planar-to-interleaved AVX-512 color conversion
;;; 4. Bit-exact equivalence checks and performance benchmarks

(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)
(asdf:load-system :sb-simd)

(in-package :jpeg)

(declaim (optimize (speed 3) (safety 0) (debug 0)))

;;; ---------------------------------------------------------------------------
;;; 1. Planar 1x1 Upsample
;;; ---------------------------------------------------------------------------

(declaim (inline upsample-1x1-planar-scalar upsample-1x1-planar-simd))

(defun upsample-1x1-planar-scalar (plane block base-pos width)
  "Write 8x8 block (sint16 clamped to [0,255]) to PLANE at base-pos with stride WIDTH."
  (declare (type (simple-array (unsigned-byte 8) (*)) plane)
           (type sint16-2d-array block)
           (type fixnum base-pos width)
           (optimize (speed 3) (safety 0)))
  (let ((pos base-pos))
    (declare (type fixnum pos))
    (dotimes (r 8)
      (let ((row (aref block r)))
        (declare (type sint16-array row))
        (setf (aref plane (+ pos 0)) (aref row 0)
              (aref plane (+ pos 1)) (aref row 1)
              (aref plane (+ pos 2)) (aref row 2)
              (aref plane (+ pos 3)) (aref row 3)
              (aref plane (+ pos 4)) (aref row 4)
              (aref plane (+ pos 5)) (aref row 5)
              (aref plane (+ pos 6)) (aref row 6)
              (aref plane (+ pos 7)) (aref row 7)))
      (incf pos width))))

(declaim (inline upsample-1x1-planar-sap upsample-2x2-planar-sap))

(defun upsample-1x1-planar-sap (sap block base-pos width)
  "Write 8x8 block to raw SAP at BASE-POS with stride WIDTH using 64-bit integer stores."
  (declare (type sb-sys:system-area-pointer sap)
           (type sint16-2d-array block)
           (type fixnum base-pos width)
           (optimize (speed 3) (safety 0)))
  (let ((pos base-pos))
    (declare (type fixnum pos))
    (dotimes (r 8)
      (let ((row (aref block r)))
        (declare (type sint16-array row))
        (let ((val (logior (the (unsigned-byte 64) (aref row 0))
                           (the (unsigned-byte 64) (ash (aref row 1) 8))
                           (the (unsigned-byte 64) (ash (aref row 2) 16))
                           (the (unsigned-byte 64) (ash (aref row 3) 24))
                           (the (unsigned-byte 64) (ash (aref row 4) 32))
                           (the (unsigned-byte 64) (ash (aref row 5) 40))
                           (the (unsigned-byte 64) (ash (aref row 6) 48))
                           (the (unsigned-byte 64) (ash (aref row 7) 56)))))
          (setf (sb-sys:sap-ref-64 sap pos) val)))
      (incf pos width))))

(defun upsample-1x1-planar-simd (plane block base-pos width)
  "Write 8x8 block to PLANE using 64-bit integer writes."
  (declare (type (simple-array (unsigned-byte 8) (*)) plane)
           (type sint16-2d-array block)
           (type fixnum base-pos width)
           (optimize (speed 3) (safety 0)))
  (sb-sys:with-pinned-objects (plane)
    (upsample-1x1-planar-sap (sb-sys:vector-sap plane) block base-pos width)))

;;; ---------------------------------------------------------------------------
;;; 2. Planar 2x2 Upsample
;;; ---------------------------------------------------------------------------

(declaim (inline upsample-2x2-planar-scalar upsample-2x2-planar-simd))

(defun upsample-2x2-planar-scalar (plane block base-pos width)
  "Upsample 8x8 chroma block by 2x2 into PLANE starting at BASE-POS with stride WIDTH."
  (declare (type (simple-array (unsigned-byte 8) (*)) plane)
           (type sint16-2d-array block)
           (type fixnum base-pos width)
           (optimize (speed 3) (safety 0)))
  (let ((row-pos base-pos)
        (width2 (* width 2)))
    (declare (type fixnum row-pos width2))
    (dotimes (r 8)
      (let ((row (aref block r))
            (p0 row-pos)
            (p1 (+ row-pos width)))
        (declare (type sint16-array row)
                 (type fixnum p0 p1))
        (macrolet ((put2 (col idx)
                     `(let ((v (aref row ,col)))
                        (setf (aref plane (+ p0 ,idx))       v
                              (aref plane (+ p0 ,(1+ idx)))   v
                              (aref plane (+ p1 ,idx))       v
                              (aref plane (+ p1 ,(1+ idx)))   v))))
          (put2 0 0)
          (put2 1 2)
          (put2 2 4)
          (put2 3 6)
          (put2 4 8)
          (put2 5 10)
          (put2 6 12)
          (put2 7 14)))
      (incf row-pos width2))))

(defun upsample-2x2-planar-sap (sap block base-pos width)
  "Upsample 8x8 chroma block to raw SAP at BASE-POS with stride WIDTH using two 64-bit writes/row."
  (declare (type sb-sys:system-area-pointer sap)
           (type sint16-2d-array block)
           (type fixnum base-pos width)
           (optimize (speed 3) (safety 0)))
  (let ((row-pos base-pos)
        (width2 (* width 2)))
    (declare (type fixnum row-pos width2))
    (dotimes (r 8)
      (let* ((row (aref block r))
             (p0 row-pos)
             (p1 (+ row-pos width))
             (c0 (aref row 0)) (c1 (aref row 1))
             (c2 (aref row 2)) (c3 (aref row 3))
             (c4 (aref row 4)) (c5 (aref row 5))
             (c6 (aref row 6)) (c7 (aref row 7))
             ;; Low 8 bytes: c0 c0 c1 c1 c2 c2 c3 c3
             (w0 (* c0 #x0101))
             (w1 (* c1 #x0101))
             (w2 (* c2 #x0101))
             (w3 (* c3 #x0101))
             (val-lo (the (unsigned-byte 64)
                          (logior w0 (ash w1 16) (ash w2 32) (ash w3 48))))
             ;; High 8 bytes: c4 c4 c5 c5 c6 c6 c7 c7
             (w4 (* c4 #x0101))
             (w5 (* c5 #x0101))
             (w6 (* c6 #x0101))
             (w7 (* c7 #x0101))
             (val-hi (the (unsigned-byte 64)
                          (logior w4 (ash w5 16) (ash w6 32) (ash w7 48)))))
        (declare (type sint16-array row)
                 (type fixnum p0 p1 c0 c1 c2 c3 c4 c5 c6 c7 w0 w1 w2 w3 w4 w5 w6 w7))
        ;; Store 16 bytes to row 0
        (setf (sb-sys:sap-ref-64 sap p0) val-lo)
        (setf (sb-sys:sap-ref-64 sap (+ p0 8)) val-hi)
        ;; Store 16 bytes to row 1
        (setf (sb-sys:sap-ref-64 sap p1) val-lo)
        (setf (sb-sys:sap-ref-64 sap (+ p1 8)) val-hi))
      (incf row-pos width2))))

(defun upsample-2x2-planar-simd (plane block base-pos width)
  "Upsample 8x8 chroma block by doubling bytes to 16 bytes across 2 rows using 64-bit stores."
  (declare (type (simple-array (unsigned-byte 8) (*)) plane)
           (type sint16-2d-array block)
           (type fixnum base-pos width)
           (optimize (speed 3) (safety 0)))
  (sb-sys:with-pinned-objects (plane)
    (upsample-2x2-planar-sap (sb-sys:vector-sap plane) block base-pos width)))

;;; ---------------------------------------------------------------------------
;;; 3. Planar-to-Interleaved AVX-512 Color Conversion
;;; ---------------------------------------------------------------------------

(defun planar-colorspace-convert-simd (dest y-plane cb-plane cr-plane npixels)
  "Convert contiguous planar Y, Cb, Cr into interleaved BGR in DEST for NPIXELS.
Processes 32 pixels per loop iteration using AVX-512BW and pinned SAPs.
Returns number of pixels processed."
  (declare (type (simple-array (unsigned-byte 8) (*)) dest y-plane cb-plane cr-plane)
           (type fixnum npixels)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((nchunks32 (floor npixels 32))
         (int-mask   (sb-simd-avx512bw:u8.64-aref +int-bgr-mask+ 0))
         (bias       (sb-simd-avx512bw:s16.32 128))
         (zero-s16   (sb-simd-avx512bw:s16.32 0))
         (max-s16    (sb-simd-avx512bw:s16.32 255))
         (zero-u8-32 (sb-simd-avx2:u8.32 0))
         (zero-u8-64 (sb-simd-avx512bw:u8.64 0))
         (k-cr-r     (sb-simd-avx512bw:s16.32 +ycc-cr-r+))
         (k-cb-g     (sb-simd-avx512bw:s16.32 +ycc-cb-g+))
         (k-cr-g     (sb-simd-avx512bw:s16.32 +ycc-cr-g+))
         (k-cb-b     (sb-simd-avx512bw:s16.32 +ycc-cb-b+)))
    (declare (type sb-simd-avx512bw:u8.64 int-mask zero-u8-64)
             (type sb-simd-avx2:u8.32 zero-u8-32)
             (type sb-simd-avx512bw:s16.32 bias zero-s16 max-s16 k-cr-r k-cb-g k-cr-g k-cb-b)
             (type fixnum nchunks32))
    (sb-sys:with-pinned-objects (dest y-plane cb-plane cr-plane)
      (let ((dest-sap (sb-sys:vector-sap dest))
            (y-sap    (sb-sys:vector-sap y-plane))
            (cb-sap   (sb-sys:vector-sap cb-plane))
            (cr-sap   (sb-sys:vector-sap cr-plane)))
        (loop for chunk fixnum from 0 below nchunks32
              for px-off fixnum = (* chunk 32)
              for dest-off fixnum = (* chunk 96) do
                (let* (;; Load 32 contiguous bytes from each plane and zero-extend directly with vpmovzxbw
                       (y32      (sb-simd-avx2:u8.32-sap-ref y-sap px-off))
                       (cb32     (sb-simd-avx2:u8.32-sap-ref cb-sap px-off))
                       (cr32     (sb-simd-avx2:u8.32-sap-ref cr-sap px-off))
                       (y16      (sb-simd-avx512bw:s16.32!
                                  (sb-simd-avx512bw:u16.32-from-u8.32 y32)))
                       (cb16     (sb-simd-avx512bw:s16.32!
                                  (sb-simd-avx512bw:u16.32-from-u8.32 cb32)))
                       (cr16     (sb-simd-avx512bw:s16.32!
                                  (sb-simd-avx512bw:u16.32-from-u8.32 cr32)))
                       ;; Cb' = Cb - 128, Cr' = Cr - 128 (32 words in parallel)
                       (cb-prime (sb-simd-avx512bw:s16.32- cb16 bias))
                       (cr-prime (sb-simd-avx512bw:s16.32- cr16 bias))
                       ;; Chroma deltas using vpmulhrsw (32 words in parallel)
                       (rd       (sb-simd-avx512bw:s16.32+
                                  cr-prime
                                  (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-r)))
                       (gd       (sb-simd-avx512bw:s16.32+
                                  (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-g)
                                  (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-g)))
                       (bd       (sb-simd-avx512bw:s16.32+
                                  cb-prime
                                  (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-b)))
                       ;; Channels: R = Y + rd, G = Y + gd, B = Y + bd
                       (r        (sb-simd-avx512bw:s16.32+ y16 rd))
                       (g        (sb-simd-avx512bw:s16.32+ y16 gd))
                       (b        (sb-simd-avx512bw:s16.32+ y16 bd))
                       ;; Clamp to [0, 255]
                       (r-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 r)))
                       (g-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 g)))
                       (b-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 b)))
                       ;; Truncate to 32 unsigned bytes (vpmovwb)
                       (rb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! r-clamp)))
                       (gb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! g-clamp)))
                       (bb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! b-clamp)))
                       ;; --- Part 1: Pixels 0..15 (low 16 bytes of B, G, R) ---
                       (b16-0    (sb-simd-avx2:u8.16-from-u8.32 bb 0))
                       (g16-0    (sb-simd-avx2:u8.16-from-u8.32 gb 0))
                       (r16-0    (sb-simd-avx2:u8.16-from-u8.32 rb 0))
                       (bg-0     (sb-simd-avx2:u8.32-insert-u8.16
                                  (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 b16-0 0)
                                  g16-0 1))
                       (r256-0   (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 r16-0 0))
                       (packed-0 (sb-simd-avx512bw:u8.64-insert-u8.32
                                  (sb-simd-avx512bw:u8.64-insert-u8.32 zero-u8-64 bg-0 0)
                                  r256-0 1))
                       (out-0    (sb-simd-avx512vbmi:u8.64-permute int-mask packed-0))
                       (low32-0  (sb-simd-avx512bw:u8.32-from-u8.64 out-0 0))
                       (mid16-0  (sb-simd-avx2:u8.16-from-u8.32 (sb-simd-avx512bw:u8.32-from-u8.64 out-0 1) 0))
                       ;; --- Part 2: Pixels 16..31 (high 16 bytes of B, G, R) ---
                       (b16-1    (sb-simd-avx2:u8.16-from-u8.32 bb 1))
                       (g16-1    (sb-simd-avx2:u8.16-from-u8.32 gb 1))
                       (r16-1    (sb-simd-avx2:u8.16-from-u8.32 rb 1))
                       (bg-1     (sb-simd-avx2:u8.32-insert-u8.16
                                  (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 b16-1 0)
                                  g16-1 1))
                       (r256-1   (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 r16-1 0))
                       (packed-1 (sb-simd-avx512bw:u8.64-insert-u8.32
                                  (sb-simd-avx512bw:u8.64-insert-u8.32 zero-u8-64 bg-1 0)
                                  r256-1 1))
                       (out-1    (sb-simd-avx512vbmi:u8.64-permute int-mask packed-1))
                       (low32-1  (sb-simd-avx512bw:u8.32-from-u8.64 out-1 0))
                       (mid16-1  (sb-simd-avx2:u8.16-from-u8.32 (sb-simd-avx512bw:u8.32-from-u8.64 out-1 1) 0)))
                  ;; Store part 1 (48 bytes)
                  (setf (sb-simd-avx2:u8.32-sap-ref dest-sap dest-off) low32-0)
                  (setf (sb-simd-sse2:u8.16-sap-ref dest-sap (+ dest-off 32)) mid16-0)
                  ;; Store part 2 (48 bytes)
                  (setf (sb-simd-avx2:u8.32-sap-ref dest-sap (+ dest-off 48)) low32-1)
                  (setf (sb-simd-sse2:u8.16-sap-ref dest-sap (+ dest-off 80)) mid16-1))))
      (* nchunks32 32))))

;;; ---------------------------------------------------------------------------
;;; 4. Tests and Verification
;;; ---------------------------------------------------------------------------

(defun test-equivalence ()
  (format t "=== 1. Testing Planar Upsample Equivalence ===~%")
  (let* ((w 64)
         (h 64)
         (interleaved (make-array (* w h 3) :element-type 'uint8 :initial-element 0))
         (plane-scalar (make-array (* w h) :element-type 'uint8 :initial-element 0))
         (plane-simd   (make-array (* w h) :element-type 'uint8 :initial-element 0))
         (block (2d-sint16-array
                 '(10  20  30  40  50  60  70  80)
                 '(15  25  35  45  55  65  75  85)
                 '(100 110 120 130 140 150 160 170)
                 '(105 115 125 135 145 155 165 175)
                 '(200 205 210 215 220 225 230 235)
                 '(201 206 211 216 221 226 231 236)
                 '(5   15  25  35  45  55  65  75)
                 '(99  88  77  66  55  44  33  22))))
    
    ;; Test 1x1
    (upsample-1x1-fast interleaved block 0 0 (* w 3))
    (upsample-1x1-planar-scalar plane-scalar block 0 w)
    (upsample-1x1-planar-simd plane-simd block 0 w)
    
    (let ((diff-scal 0) (diff-simd 0))
      (dotimes (r 8)
        (dotimes (c 8)
          (let ((int-val (aref interleaved (+ (* r w 3) (* c 3))))
                (scal-val (aref plane-scalar (+ (* r w) c)))
                (simd-val (aref plane-simd (+ (* r w) c))))
            (unless (= int-val scal-val) (incf diff-scal))
            (unless (= int-val simd-val) (incf diff-simd)))))
      (format t "  1x1 Planar Scalar diffs: ~D~%" diff-scal)
      (format t "  1x1 Planar SIMD   diffs: ~D~%" diff-simd)
      (assert (zerop diff-scal))
      (assert (zerop diff-simd)))

    ;; Test 2x2
    (fill interleaved 0)
    (fill plane-scalar 0)
    (fill plane-simd 0)
    (upsample-2x2-fast interleaved block 1 0 (* w 3))
    (upsample-2x2-planar-scalar plane-scalar block 0 w)
    (upsample-2x2-planar-simd plane-simd block 0 w)

    (let ((diff-scal 0) (diff-simd 0))
      (dotimes (r 16)
        (dotimes (c 16)
          (let ((int-val (aref interleaved (+ (* r w 3) (* c 3) 1)))
                (scal-val (aref plane-scalar (+ (* r w) c)))
                (simd-val (aref plane-simd (+ (* r w) c))))
            (unless (= int-val scal-val) (incf diff-scal))
            (unless (= int-val simd-val) (incf diff-simd)))))
      (format t "  2x2 Planar Scalar diffs: ~D~%" diff-scal)
      (format t "  2x2 Planar SIMD   diffs: ~D~%" diff-simd)
      (assert (zerop diff-scal))
      (assert (zerop diff-simd)))
    (format t "  => ALL PLANAR UPSAMPLE TESTS PASSED BIT-EXACT!~%~%"))

  (format t "=== 2. Testing Planar Color Conversion Equivalence ===~%")
  (let* ((w 64)
         (h 64)
         (npixels (* w h))
         (dest-ref (make-array (* npixels 3) :element-type 'uint8 :initial-element 0))
         (dest-int-simd (make-array (+ (* npixels 3) 64) :element-type 'uint8 :initial-element 0))
         (dest-pln-simd (make-array (+ (* npixels 3) 64) :element-type 'uint8 :initial-element 0))
         (y-plane (make-array (+ npixels 64) :element-type 'uint8 :initial-element 0))
         (cb-plane (make-array (+ npixels 64) :element-type 'uint8 :initial-element 0))
         (cr-plane (make-array (+ npixels 64) :element-type 'uint8 :initial-element 0)))
    ;; Populate test pattern
    (dotimes (i npixels)
      (let ((y (mod (+ i 50) 256))
            (cb (mod (* i 7) 256))
            (cr (mod (* i 13) 256)))
        (setf (aref y-plane i) y
              (aref cb-plane i) cb
              (aref cr-plane i) cr
              ;; Interleaved ref
              (aref dest-ref (+ (* i 3) 0)) y
              (aref dest-ref (+ (* i 3) 1)) cb
              (aref dest-ref (+ (* i 3) 2)) cr
              ;; Interleaved SIMD
              (aref dest-int-simd (+ (* i 3) 0)) y
              (aref dest-int-simd (+ (* i 3) 1)) cb
              (aref dest-int-simd (+ (* i 3) 2)) cr)))
    ;; Reference scalar conversion on dest-ref
    (loop for px fixnum from 0 below npixels
          for py fixnum = (* px 3)
          for pu fixnum = (1+ py)
          for pv fixnum = (+ py 2)
          for yy fixnum = (aref dest-ref py)
          for cb fixnum = (aref dest-ref pu)
          for cr fixnum = (aref dest-ref pv) do
            (setf (aref dest-ref py) (the uint8 (limit (+ yy (aref +cb-b-tab+ cb)))))
            (setf (aref dest-ref pu) (the uint8 (limit (+ yy (ash (+ (aref +cb-g-tab+ cb)
                                                                    (aref +cr-g-tab+ cr))
                                                                 (- shift))))))
            (setf (aref dest-ref pv) (the uint8 (limit (+ yy (aref +cr-r-tab+ cr))))))
    ;; Run Interleaved SIMD
    (inverse-colorspace-convert-simd dest-int-simd npixels)
    ;; Run Planar SIMD
    (planar-colorspace-convert-simd dest-pln-simd y-plane cb-plane cr-plane npixels)

    ;; 1. Check bit-exact identity with existing SIMD kernel
    (let ((diffs-vs-simd 0))
      (dotimes (i (* npixels 3))
        (unless (= (aref dest-int-simd i) (aref dest-pln-simd i))
          (incf diffs-vs-simd)))
      (format t "  Planar SIMD vs Interleaved SIMD diffs: ~D / ~D~%" diffs-vs-simd (* npixels 3))
      (assert (zerop diffs-vs-simd)))

    ;; 2. Check maximum difference vs scalar reference (<= 1 LSB)
    (let ((max-diff 0)
          (mismatches 0))
      (dotimes (i (* npixels 3))
        (let ((d (abs (- (aref dest-pln-simd i) (aref dest-ref i)))))
          (when (> d 0) (incf mismatches))
          (when (> d max-diff) (setf max-diff d))))
      (format t "  Planar SIMD vs Scalar max-diff: ~D LSB (exact matches: ~,1F%)~%"
              max-diff (* 100.0 (/ (- (* npixels 3) mismatches) (* npixels 3))))
      (assert (<= max-diff 1)))
    (format t "  => PLANAR COLOR CONVERSION PASSED BIT-EXACT IDENTITY WITH INTERLEAVED SIMD!~%~%")))

(defun run-benchmarks ()
  (format t "=== 3. Performance Microbenchmarks (100,000 calls) ===~%")
  (let* ((w 1024)
         (h 1024)
         (nwidth (* w 3))
         (int-buf (make-array (* w h 3) :element-type 'uint8 :initial-element 0))
         (plane   (make-array (* w h)   :element-type 'uint8 :initial-element 0))
         (block (2d-sint16-array
                 '(128 128 128 128 128 128 128 128)
                 '(128 128 128 128 128 128 128 128)
                 '(128 128 128 128 128 128 128 128)
                 '(128 128 128 128 128 128 128 128)
                 '(128 128 128 128 128 128 128 128)
                 '(128 128 128 128 128 128 128 128)
                 '(128 128 128 128 128 128 128 128)
                 '(128 128 128 128 128 128 128 128)))
         (n-trials 100000))
    ;; 1x1
    (let ((t0 (get-internal-real-time)))
      (dotimes (i n-trials)
        (upsample-1x1-fast int-buf block 0 0 nwidth))
      (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "  Current upsample-1x1-fast:      ~,2F ms (~,2F ns/call)~%" (* dt 1000) (/ (* dt 1e9) n-trials))))

    (let ((t0 (get-internal-real-time)))
      (dotimes (i n-trials)
        (upsample-1x1-planar-scalar plane block 0 w))
      (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "  Planar 1x1 Scalar:             ~,2F ms (~,2F ns/call)~%" (* dt 1000) (/ (* dt 1e9) n-trials))))

    (let ((t0 (get-internal-real-time)))
      (sb-sys:with-pinned-objects (plane)
        (let ((sap (sb-sys:vector-sap plane)))
          (dotimes (i n-trials)
            (upsample-1x1-planar-sap sap block 0 w))))
      (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "  Planar 1x1 SAP (pinned outside): ~,2F ms (~,2F ns/call)~%" (* dt 1000) (/ (* dt 1e9) n-trials))))

    ;; 2x2
    (let ((t0 (get-internal-real-time)))
      (dotimes (i n-trials)
        (upsample-2x2-fast int-buf block 1 0 nwidth))
      (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "  Current upsample-2x2-fast:      ~,2F ms (~,2F ns/call)~%" (* dt 1000) (/ (* dt 1e9) n-trials))))

    (let ((t0 (get-internal-real-time)))
      (dotimes (i n-trials)
        (upsample-2x2-planar-scalar plane block 0 w))
      (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "  Planar 2x2 Scalar:             ~,2F ms (~,2F ns/call)~%" (* dt 1000) (/ (* dt 1e9) n-trials))))

    (let ((t0 (get-internal-real-time)))
      (sb-sys:with-pinned-objects (plane)
        (let ((sap (sb-sys:vector-sap plane)))
          (dotimes (i n-trials)
            (upsample-2x2-planar-sap sap block 0 w))))
      (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
        (format t "  Planar 2x2 SAP (pinned outside): ~,2F ms (~,2F ns/call)~%" (* dt 1000) (/ (* dt 1e9) n-trials))))

    ;; Color Conversion 1024x1024
    (format t "~%=== 4. Color Conversion Benchmark 1024x1024 (100 reps) ===~%")
    (let* ((npixels (* 1024 1024))
           (dest-int (make-array (+ (* npixels 3) 64) :element-type 'uint8 :initial-element 128))
           (dest-pln (make-array (+ (* npixels 3) 64) :element-type 'uint8 :initial-element 128))
           (y-pln    (make-array (+ npixels 64) :element-type 'uint8 :initial-element 128))
           (cb-pln   (make-array (+ npixels 64) :element-type 'uint8 :initial-element 128))
           (cr-pln   (make-array (+ npixels 64) :element-type 'uint8 :initial-element 128))
           (reps 100))
      ;; Interleaved SIMD color conv
      (let ((t0 (get-internal-real-time)))
        (dotimes (i reps)
          (inverse-colorspace-convert-simd dest-int npixels))
        (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
          (format t "  Interleaved SIMD (de-int + conv + re-int): ~,2F ms (~,2F ms/call)~%"
                  (* dt 1000) (/ (* dt 1000) reps))))
      ;; Planar SIMD color conv
      (let ((t0 (get-internal-real-time)))
        (dotimes (i reps)
          (planar-colorspace-convert-simd dest-pln y-pln cb-pln cr-pln npixels))
        (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
          (format t "  Planar SIMD (direct load + conv + re-int):  ~,2F ms (~,2F ms/call)~%"
                  (* dt 1000) (/ (* dt 1000) reps)))))))

;;; ---------------------------------------------------------------------------
;;; 5. Full End-to-End Planar Decoder Prototype
;;; ---------------------------------------------------------------------------

(defvar *proto-y-plane* nil)
(defvar *proto-cb-plane* nil)
(defvar *proto-cr-plane* nil)

(defun proto-planar-upsample (image scan block x y H V offset nwidth nw nx dend)
  (declare (ignore nwidth nw nx dend)
           (type sint16-2d-array block)
           (type fixnum x y H V offset)
           (optimize (speed 3) (safety 0)))
  (let* ((xbase (+ (scan-x scan) x))
         (ybase (+ (scan-y scan) y))
         (width (descriptor-width image))
         (height (descriptor-height image))
         (base-pos (+ (* ybase width) xbase))
         (plane (the (simple-array (unsigned-byte 8) (*))
                     (cond ((= offset 0) *proto-y-plane*)
                           ((= offset 1) *proto-cb-plane*)
                           (t *proto-cr-plane*)))))
    (declare (type fixnum xbase ybase width height base-pos))
    (cond
      ;; 1x1 fast path (H=1, V=1)
      ((and (= H 1) (= V 1) (<= (+ xbase 8) width) (<= (+ ybase 8) height))
       (upsample-1x1-planar-scalar plane block base-pos width))
      
      ;; 2x2 fast path (H=2, V=2)
      ((and (= H 2) (= V 2) (<= (+ xbase 16) width) (<= (+ ybase 16) height))
       (upsample-2x2-planar-scalar plane block base-pos width))
      
      ;; Fallback for boundaries or non-standard sampling
      (t
       (loop for row of-type sint16-array across block
             for y-idx fixnum from ybase below height by V
             for ypos fixnum from (* ybase width) by (* width V) do
               (loop for val fixnum across row
                     for x-idx fixnum from xbase below width by H
                     for pos fixnum from (+ ypos xbase) by H do
                       (if (= 1 H V)
                           (setf (aref plane pos) val)
                           (loop for dy fixnum from 0 below V
                                 for dypos fixnum from pos by width
                                 for dxend fixnum = (+ ypos (* (+ dy 1) width)) do
                                   (loop for dx fixnum from 0 below H
                                         for dpos fixnum from dypos below dxend do
                                           (when (< dpos (length plane))
                                             (setf (aref plane dpos) val)))))))))))

(defun decode-image-planar (path)
  "Decode JPEG at PATH using Planar Chroma Intermediate and AVX-512 Planar Color Conversion."
  (multiple-value-bind (height width ncomp)
      (jpeg-file-dimensions path)
    (let* ((npixels (* width height))
           (y-plane  (make-array (+ npixels 64) :element-type 'uint8 :initial-element 0))
           (cb-plane (make-array (+ npixels 64) :element-type 'uint8 :initial-element 0))
           (cr-plane (make-array (+ npixels 64) :element-type 'uint8 :initial-element 0))
           (*proto-y-plane* y-plane)
           (*proto-cb-plane* cb-plane)
           (*proto-cr-plane* cr-plane)
           (orig-upsample (fdefinition 'upsample)))
      ;; Hook upsample to write to planar buffers
      (setf (fdefinition 'upsample) #'proto-planar-upsample)
      (unwind-protect
           (let ((buf (decode-image path :colorspace-conversion nil)))
             (when (and (= ncomp 3) *simd-decode-color-p*)
               (let* ((done (planar-colorspace-convert-simd buf y-plane cb-plane cr-plane npixels)))
                 (when (< done npixels)
                   (inverse-colorspace-convert-scalar-tail buf done npixels))))
             buf)
        ;; Restore original upsample
        (setf (fdefinition 'upsample) orig-upsample)))))

(defun test-full-image (path name)
  (format t "Testing full-image decode: ~A (~A)...~%" name path)
  (let* ((ref-buf (decode-image path))
         (pln-buf (decode-image-planar path))
         (diffs 0)
         (max-diff 0))
    (dotimes (i (length ref-buf))
      (let ((d (abs (- (aref ref-buf i) (aref pln-buf i)))))
        (when (> d 0)
          (incf diffs)
          (when (> d max-diff) (setf max-diff d)))))
    (format t "  Result: ~D / ~D mismatches, max-diff = ~D LSB~%" diffs (length ref-buf) max-diff)
    (assert (zerop diffs))
    (format t "  => ~A PASSED 100% BIT-EXACT!~%~%" name)))

(defun bench-full-image (path name reps)
  (format t "Benchmarking ~A (~:D reps)...~%" name reps)
  ;; Warmup
  (dotimes (i 5)
    (decode-image path)
    (decode-image-planar path))
  ;; Baseline (current full SIMD)
  (sb-ext:gc :full t)
  (let ((t0 (get-internal-real-time)))
    (dotimes (i reps)
      (decode-image path))
    (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
      (format t "  Current Full SIMD Decoder: ~,2F ms (~,2F ms/image)~%"
              (* dt 1000) (/ (* dt 1000) reps))))
  ;; Planar Intermediate Decoder
  (sb-ext:gc :full t)
  (let ((t0 (get-internal-real-time)))
    (dotimes (i reps)
      (decode-image-planar path))
    (let ((dt (/ (- (get-internal-real-time) t0) (float internal-time-units-per-second))))
      (format t "  Planar Intermediate Decoder: ~,2F ms (~,2F ms/image)~%"
              (* dt 1000) (/ (* dt 1000) reps)))))

;;; End of proto-planar-upsample.lisp
