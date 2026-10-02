;;;; simd.lisp --- AVX-512 accelerated kernels for CL-JPEG
;;;;
;;;; Phase 1A: decoder YCbCr -> RGB inverse color conversion.
;;;;
;;;; Target: AMD Zen 5 (AVX512F + AVX512BW + AVX512VBMI).
;;;;
;;;; Because the scalar path uses 55-bit fixed-point lookup tables while
;;;; this SIMD path uses 15-bit vpmulhrsw arithmetic, the two paths differ
;;;; by +/- 1 LSB on approximately 8.3% of pixels (where two-step vs
;;;; one-step rounding straddles a rounding boundary). Both paths are
;;;; fully JFIF-compliant (chroma zero bias 128). Do not attempt to eliminate
;;;; this difference; doing so would require either slower 32-bit SIMD arithmetic
;;;; or matching the scalar table rounding exactly (not expressible in vpmulhrsw).
;;;;
;;;; Architectural principle: STORE-FORWARDING AVOIDANCE
;;;; Loads and stores in adjacent iterations must not overlap in address space.
;;;; On Zen 5, a 64-byte load that partially overlaps a pending narrower store
;;;; triggers a store-forwarding stall that can cost more than 2x. Writing exactly
;;;; 48 bytes per 16-pixel chunk (32-byte + 16-byte stores) completely avoids
;;;; touching the bytes loaded by the next iteration.
;;;;
;;;; The kernel processes 16 pixels per iteration. Input is a flat
;;;; interleaved [Y Cb Cr] byte buffer (3 bytes/pixel). Output is the
;;;; same buffer, rewritten in place as interleaved [B G R].

(in-package #:jpeg)

#+(and sbcl x86-64)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (handler-case
      (require :sb-simd)
    (error () nil)))

;;; Availability

(declaim (type fixnum *simd-color-chunks-processed*))
(defvar *simd-color-chunks-processed* 0
  "Total number of 16-pixel SIMD chunks processed by the color conversion kernel.")

(defvar *simd-decode-color-p* nil
  "True when the AVX-512 decoder color-conversion kernel is usable.")

(defvar *simd-encode-dct-p* nil
  "True when the AVX2 forward DCT kernel is usable.")

(defvar *simd-decode-dct-p* nil
  "True when the AVX2 inverse DCT kernel is usable.")

(defvar *simd-planar-upsample-p* t
  "True when planar intermediate upsampling is usable.")

(defvar *fused-chroma-dup-mask*
  (let ((arr (make-array 64 :element-type '(unsigned-byte 8)
                            :initial-contents (loop for i from 0 below 64 collect (if (< i 32) (floor i 2) 0)))))
    #+(and sbcl x86-64)
    (sb-simd-avx512bw:u8.64-aref arr 0)
    #-(and sbcl x86-64)
    nil)
  "AVX-512VBMI permute mask that duplicates 16 bytes of chroma into 32 bytes (indices 0,0,1,1,...,15,15).")

(eval-when (:load-toplevel :execute)
  (setf *simd-decode-color-p*
        (handler-case
            #+(and sbcl x86-64)
            (and (find-package "SB-SIMD-AVX512F")
                 (find-package "SB-SIMD-AVX512BW")
                 (find-package "SB-SIMD-AVX512VBMI")
                 (sb-simd-internals:instruction-set-available-p
                  (sb-simd-internals:find-instruction-set :avx512f))
                 (sb-simd-internals:instruction-set-available-p
                  (sb-simd-internals:find-instruction-set :avx512bw))
                 (sb-simd-internals:instruction-set-available-p
                  (sb-simd-internals:find-instruction-set :avx512vbmi)))
            #-(and sbcl x86-64)
            nil
          (error () nil)))
  (setf *simd-encode-dct-p*
        (handler-case
            #+(and sbcl x86-64)
            (and (find-package "SB-SIMD-AVX2")
                 (sb-simd-internals:instruction-set-available-p
                  (sb-simd-internals:find-instruction-set :avx2)))
            #-(and sbcl x86-64)
            nil
          (error () nil)))
  (setf *simd-decode-dct-p*
        (handler-case
            #+(and sbcl x86-64)
            (and (find-package "SB-SIMD-AVX2")
                 (sb-simd-internals:instruction-set-available-p
                  (sb-simd-internals:find-instruction-set :avx2)))
            #-(and sbcl x86-64)
            nil
          (error () nil))))


;;; Fixed-point coefficients
;;;
;;;   R = Y                  + 1.40200 * Cr' = Y + Cr' + 0.40200 * Cr'
;;;   G = Y - 0.34414 * Cb'  - 0.71414 * Cr'
;;;   B = Y                  + 1.77200 * Cb' = Y + Cb' + 0.77200 * Cb'
;;;
;;; with Cb' = Cb - 128, Cr' = Cr - 128.
;;;
;;; vpmulhrsw computes (a * b + 0x4000) >> 15 for signed 16-bit a, b.
;;; The constants below are scaled by 2^15 so that vpmulhrsw produces the
;;; correctly rounded product. For R and B, factoring out 1.0 eliminates
;;; 16-bit signed overflow and preserves precision.

#+(and sbcl x86-64)
(defconstant +ycc-cr-r+  13173)  ; round( 0.40200 * 2^15)
#+(and sbcl x86-64)
(defconstant +ycc-cb-g+ -11277)  ; round(-0.34414 * 2^15)
#+(and sbcl x86-64)
(defconstant +ycc-cr-g+ -23400)  ; round(-0.71414 * 2^15)
#+(and sbcl x86-64)
(defconstant +ycc-cb-b+  25297)  ; round( 0.77200 * 2^15)

;;; Permute masks for de-interleaving and interleaving
;;;
;;; Source layout (48 bytes of a 64-byte load), where 3k..3k+2 = pixel k:
;;;   byte  3k+0 : Y_k
;;;   byte  3k+1 : Cb_k
;;;   byte  3k+2 : Cr_k
;;;
;;; Single-channel masks gather 16 bytes for a given channel into bytes 0..15.
;;; Bytes 16..63 are routed to index 0 (harmless because they are masked out).
;;;
;;; +int-bgr-mask+ consumes [B | G | R] packed into bytes 0..47 of a 64-byte
;;; register (B in 0..15, G in 16..31, R in 32..47) and produces interleaved
;;; [B G R] across bytes 0..47.

#+(and sbcl x86-64)
(declaim (type (simple-array (unsigned-byte 8) (64))
               +deint-y-only-mask+
               +deint-cb-only-mask+
               +deint-cr-only-mask+
               +int-bgr-mask+))

#+(and sbcl x86-64)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun make-single-channel-mask (offset)
    "Mask that gathers 16 bytes at stride 3 starting at OFFSET, into
bytes 0..15 of the destination; remaining bytes routed to index 0."
    (make-array 64 :element-type '(unsigned-byte 8)
                :initial-contents
                (append (loop for k from 0 below 16 collect (+ offset (* 3 k)))
                        (make-list 48 :initial-element 0))))

  (defparameter +deint-y-only-mask+  (make-single-channel-mask 0))
  (defparameter +deint-cb-only-mask+ (make-single-channel-mask 1))
  (defparameter +deint-cr-only-mask+ (make-single-channel-mask 2))

  (defparameter +int-bgr-mask+
    (make-array 64 :element-type '(unsigned-byte 8)
                :initial-contents
                (append
                 (loop for k from 0 below 16
                       append (list k (+ 16 k) (+ 32 k)))
                 (make-list 16 :initial-element 0)))))

;;; SIMD Kernel
;;;
;;; Processes floor(npixels/16)*16 pixels starting at byte offset 0 of
;;; BUFFER. Returns the number of pixels actually processed.
;;;
;;; The caller is responsible for the trailing (npixels mod 16) pixels,
;;; and must guarantee that BUFFER has at least 16 bytes of valid storage
;;; beyond the last processed chunk (so the 64-byte load does not overrun).

#+(and sbcl x86-64)
(defun inverse-colorspace-convert-simd (buffer npixels)
  "In-place YCbCr -> RGB conversion on BUFFER (flat u8 array of interleaved
[Y Cb Cr] bytes). Processes floor(NPIXELS/16)*16 pixels. Returns the
number of pixels processed."
  (declare (type (simple-array (unsigned-byte 8) (*)) buffer)
           (type fixnum npixels)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((nchunks (floor npixels 16))
         (y-mask   (sb-simd-avx512bw:u8.64-aref +deint-y-only-mask+ 0))
         (cb-mask  (sb-simd-avx512bw:u8.64-aref +deint-cb-only-mask+ 0))
         (cr-mask  (sb-simd-avx512bw:u8.64-aref +deint-cr-only-mask+ 0))
         (int-mask (sb-simd-avx512bw:u8.64-aref +int-bgr-mask+ 0))
         (bias     (sb-simd-avx512bw:s16.32 128))
         (zero-s16 (sb-simd-avx512bw:s16.32 0))
         (max-s16  (sb-simd-avx512bw:s16.32 255))
         (zero-u8-32 (sb-simd-avx2:u8.32 0))
         (zero-u8-64 (sb-simd-avx512bw:u8.64 0))
         (k-cr-r   (sb-simd-avx512bw:s16.32 +ycc-cr-r+))
         (k-cb-g   (sb-simd-avx512bw:s16.32 +ycc-cb-g+))
         (k-cr-g   (sb-simd-avx512bw:s16.32 +ycc-cr-g+))
         (k-cb-b   (sb-simd-avx512bw:s16.32 +ycc-cb-b+)))
    (declare (type sb-simd-avx512bw:u8.64 y-mask cb-mask cr-mask int-mask zero-u8-64)
             (type sb-simd-avx2:u8.32 zero-u8-32)
             (type sb-simd-avx512bw:s16.32 bias zero-s16 max-s16 k-cr-r k-cb-g k-cr-g k-cb-b)
             (type fixnum nchunks))
    (incf *simd-color-chunks-processed* nchunks)
    (loop for chunk fixnum from 0 below nchunks
          for byte-off fixnum = (* chunk 48) do
            (let* ((in       (sb-simd-avx512bw:u8.64-aref buffer byte-off))
                   ;; De-interleave using vpermb: arg0 is mask, arg1 is data
                   (y-perm   (sb-simd-avx512vbmi:u8.64-permute y-mask in))
                   (cb-perm  (sb-simd-avx512vbmi:u8.64-permute cb-mask in))
                   (cr-perm  (sb-simd-avx512vbmi:u8.64-permute cr-mask in))
                   ;; Zero-extend low 16 bytes to 16-bit words (vpmovzxbw)
                   (y16      (sb-simd-avx512bw:s16.32!
                              (sb-simd-avx512bw:u16.32-from-u8.32
                               (sb-simd-avx512bw:u8.32-from-u8.64 y-perm 0))))
                   (cb16     (sb-simd-avx512bw:s16.32!
                              (sb-simd-avx512bw:u16.32-from-u8.32
                               (sb-simd-avx512bw:u8.32-from-u8.64 cb-perm 0))))
                   (cr16     (sb-simd-avx512bw:s16.32!
                              (sb-simd-avx512bw:u16.32-from-u8.32
                               (sb-simd-avx512bw:u8.32-from-u8.64 cr-perm 0))))
                   ;; Cb' = Cb - 128, Cr' = Cr - 128
                   (cb-prime (sb-simd-avx512bw:s16.32- cb16 bias))
                   (cr-prime (sb-simd-avx512bw:s16.32- cr16 bias))
                   ;; Chroma deltas using vpmulhrsw:
                   ;; rd = Cr' + (Cr' * 13173 + 16384) >> 15
                   (rd       (sb-simd-avx512bw:s16.32+
                              cr-prime
                              (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-r)))
                   ;; gd = (Cb' * -11277 + 16384) >> 15 + (Cr' * -23400 + 16384) >> 15
                   (gd       (sb-simd-avx512bw:s16.32+
                              (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-g)
                              (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-g)))
                   ;; bd = Cb' + (Cb' * 25297 + 16384) >> 15
                   (bd       (sb-simd-avx512bw:s16.32+
                              cb-prime
                              (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-b)))
                   ;; Channels: R = Y + rd, G = Y + gd, B = Y + bd
                   (r        (sb-simd-avx512bw:s16.32+ y16 rd))
                   (g        (sb-simd-avx512bw:s16.32+ y16 gd))
                   (b        (sb-simd-avx512bw:s16.32+ y16 bd))
                   ;; Clamp to [0, 255] with vpmaxsw / vpminsw
                   (r-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 r)))
                   (g-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 g)))
                   (b-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 b)))
                   ;; Truncate to u8 with vpmovwb
                   (rb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! r-clamp)))
                   (gb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! g-clamp)))
                   (bb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! b-clamp)))
                   ;; Pack 16-byte B, G, R results into 64-byte register
                   (b16      (sb-simd-avx2:u8.16-from-u8.32 bb 0))
                   (g16      (sb-simd-avx2:u8.16-from-u8.32 gb 0))
                   (r16      (sb-simd-avx2:u8.16-from-u8.32 rb 0))
                   (bg       (sb-simd-avx2:u8.32-insert-u8.16
                              (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 b16 0)
                              g16 1))
                   (r256     (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 r16 0))
                   (packed   (sb-simd-avx512bw:u8.64-insert-u8.32
                              (sb-simd-avx512bw:u8.64-insert-u8.32 zero-u8-64 bg 0)
                              r256 1))
                   ;; Interleave with vpermb: byte 3k=B_k, 3k+1=G_k, 3k+2=R_k
                   (out      (sb-simd-avx512vbmi:u8.64-permute int-mask packed))
                   ;; Write exactly 48 bytes without overrunning (32-byte + 16-byte stores)
                   ;; This avoids store-forwarding aliased stalls with subsequent chunk loads.
                   (low32    (sb-simd-avx512bw:u8.32-from-u8.64 out 0))
                   (mid16    (sb-simd-avx2:u8.16-from-u8.32 (sb-simd-avx512bw:u8.32-from-u8.64 out 1) 0)))
              (setf (sb-simd-avx2:u8.32-aref buffer byte-off) low32)
              (setf (sb-simd-sse2:u8.16-aref buffer (+ byte-off 32)) mid16)))
    (* nchunks 16)))

;;; Scalar Tail Fallback
;;;
;;; Operates on BUFFER between start-pixel and end-pixel.

(defun inverse-colorspace-convert-scalar-tail (buffer start-pixel end-pixel)
  "Scalar fallback for pixels in the range [start-pixel, end-pixel)."
  (declare (type (simple-array (unsigned-byte 8) (*)) buffer)
           (type fixnum start-pixel end-pixel)
           (optimize (speed 3) (safety 0)))
  (loop for px fixnum from start-pixel below end-pixel
        for py fixnum = (* px 3)
        for pu fixnum = (1+ py)
        for pv fixnum = (+ py 2)
        for yy fixnum = (aref buffer py)
        for cb fixnum = (aref buffer pu)
        for cr fixnum = (aref buffer pv) do
          (setf (aref buffer py) ; BLUE
                (the (unsigned-byte 8) (limit (+ yy (aref +cb-b-tab+ cb)))))
          (setf (aref buffer pu) ; GREEN
                (the (unsigned-byte 8) (limit (+ yy (ash (+ (aref +cb-g-tab+ cb)
                                                            (aref +cr-g-tab+ cr))
                                                         (- shift))))))
          (setf (aref buffer pv) ; RED
                (the (unsigned-byte 8) (limit (+ yy (aref +cr-r-tab+ cr)))))))

;;; Fused Planar Chroma Architecture
;;;
;;; In this architecture:
;;; 1. Y is stored at full resolution (W x H).
;;; 2. Cb and Cr are stored at native reduced resolution (quarter size for 4:2:0: Wc x Hc).
;;; 3. During MCU decode, PLANAR-UPSAMPLE writes 8x8 blocks directly into native coordinates:
;;;    Y at (x, y) with stride W; Cb/Cr at (x>>1, y>>1) with stride Wc. No separate 2x2
;;;    chroma replication pass is performed.
;;; 4. At frame completion, FUSED-420-COLORSPACE-CONVERT-SIMD reads Y, Cb, and Cr at their
;;;    native resolutions, duplicates chroma in-register via AVX-512 VPERMB, and writes
;;;    interleaved BGR pixels directly.
;;; Total intermediate memory: W*H + 2*(W*H/4) + 3*W*H = 4.5*W*H bytes (vs 6*W*H previously).

(declaim (inline upsample-1x1-planar-fast))

(defun upsample-1x1-planar-fast (plane block base-pos img-w)
  "Write 8x8 block to PLANE at BASE-POS with stride IMG-W.
Precondition: The caller must ensure that (+ base-pos (* 7 img-w) 8) is within bounds,
i.e., (+ xbase 8) <= img-w and (+ ybase 8) <= img-h. No tail guards are performed."
  (declare (type (simple-array (unsigned-byte 8) (*)) plane)
           (type sint16-2d-array block)
           (type fixnum base-pos img-w)
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
      (incf pos img-w))))

(defun planar-upsample (image scan block blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)
  "Upsample a decoded 8x8 block into planar storage (Y, Cb, or Cr plane).
Planar decoder path: index 0=Y, 1=Cb, 2=Cr, selecting a plane.
(Note: interleaved output byte order is byte 0=B, 1=G, 2=R; the indices coincide
only in the 3-component JFIF case)."
  (declare (optimize (speed 3) (safety 0))
           (type sint16-2d-array block)
           (type fixnum blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)
           (ignore nw nx dend))
  (let* ((plane (cond ((= comp-index 0) (descriptor-y-plane image))
                      ((= comp-index 1) (descriptor-cb-plane image))
                      ((= comp-index 2) (descriptor-cr-plane image))
                      (t nil)))
         (xbase (the fixnum (+ (the fixnum (scan-x scan)) blk-x)))
         (ybase (the fixnum (+ (the fixnum (scan-y scan)) blk-y)))
         (img-w (descriptor-width image))
         (img-h (descriptor-height image)))
    (declare (type fixnum xbase ybase img-w img-h))
    (unless plane
      (return-from planar-upsample
        (upsample-default image scan block blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)))
    (cond
      ;; 1. Luma (comp-index = 0) at full resolution
      ((= comp-index 0)
       (if (and (<= (+ xbase 8) img-w) (<= (+ ybase 8) img-h))
           (upsample-1x1-planar-fast plane block (+ (* ybase img-w) xbase) img-w)
           (when (and (< xbase img-w) (< ybase img-h))
             (let ((max-r (min 8 (- img-h ybase)))
                   (max-c (min 8 (- img-w xbase))))
               (declare (type fixnum max-r max-c))
               (dotimes (r max-r)
                 (let ((row (aref block r))
                       (pos (+ (* (+ ybase r) img-w) xbase)))
                   (declare (type sint16-array row) (type fixnum pos))
                   (dotimes (c max-c)
                     (setf (aref plane (+ pos c)) (aref row c)))))))))

      ;; 2. 4:2:0 Chroma Cb (comp-index = 1) or Cr (comp-index = 2) at native quarter resolution
      ((and (or (= comp-index 1) (= comp-index 2)) (= samp-h 2) (= samp-v 2))
       (let* ((chroma-w (the fixnum (ash (the fixnum (1+ img-w)) -1)))
              (chroma-h (the fixnum (ash (the fixnum (1+ img-h)) -1)))
              ;; Convert MCU luma position to native chroma coordinates
              (xbase-c (the fixnum (ash xbase -1)))
              (ybase-c (the fixnum (ash ybase -1))))
         (declare (type fixnum chroma-w chroma-h xbase-c ybase-c))
         (if (and (<= (+ xbase-c 8) chroma-w) (<= (+ ybase-c 8) chroma-h))
             (upsample-1x1-planar-fast plane block (+ (* ybase-c chroma-w) xbase-c) chroma-w)
             ;; Edge-rounding note on odd-dimension boundaries:
             ;; For images with odd dimensions (e.g. testorig 227x149), integer truncation at boundaries
             ;; (such as (ash xbase -1)) can produce +/-1 LSB difference on edge pixels compared to
             ;; interleaved clamp replication. This is an expected edge-rounding artifact, not an algorithm error.
             (when (and (< xbase-c chroma-w) (< ybase-c chroma-h))
               (let ((max-r (min 8 (- chroma-h ybase-c)))
                     (max-c (min 8 (- chroma-w xbase-c))))
                 (declare (type fixnum max-r max-c))
                 (dotimes (r max-r)
                   (let ((row (aref block r))
                         (pos (+ (* (+ ybase-c r) chroma-w) xbase-c)))
                     (declare (type sint16-array row) (type fixnum pos))
                     (dotimes (c max-c)
                       (setf (aref plane (+ pos c)) (aref row c))))))))))

      ;; 3. 4:4:4 Chroma (H=1, V=1) at full resolution
      ((and (= samp-h 1) (= samp-v 1))
       (if (and (<= (+ xbase 8) img-w) (<= (+ ybase 8) img-h))
           (upsample-1x1-planar-fast plane block (+ (* ybase img-w) xbase) img-w)
           (when (and (< xbase img-w) (< ybase img-h))
             (let ((max-r (min 8 (- img-h ybase)))
                   (max-c (min 8 (- img-w xbase))))
               (declare (type fixnum max-r max-c))
               (dotimes (r max-r)
                 (let ((row (aref block r))
                       (pos (+ (* (+ ybase r) img-w) xbase)))
                   (declare (type sint16-array row) (type fixnum pos))
                   (dotimes (c max-c)
                     (setf (aref plane (+ pos c)) (aref row c)))))))))

      ;; Fallback
      (t
       (upsample-default image scan block blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)))))

(defun planar-colorspace-convert-scalar-tail (dest y-plane cb-plane cr-plane start end)
  (declare (type (simple-array (unsigned-byte 8) (*)) dest y-plane cb-plane cr-plane)
           (type fixnum start end)
           (optimize (speed 3) (safety 0)))
  (loop for i fixnum from start below end
        for dest-idx fixnum = (* i 3)
        for yy fixnum = (aref y-plane i)
        for cb fixnum = (aref cb-plane i)
        for cr fixnum = (aref cr-plane i) do
        (setf (aref dest dest-idx)
              (the uint8 (limit (+ yy (aref +cb-b-tab+ cb)))))
        (setf (aref dest (+ dest-idx 1))
              (the uint8 (limit (+ yy (ash (+ (aref +cb-g-tab+ cb)
                                              (aref +cr-g-tab+ cr))
                                           (- shift))))))
        (setf (aref dest (+ dest-idx 2))
              (the uint8 (limit (+ yy (aref +cr-r-tab+ cr)))))))

(defun fused-420-colorspace-convert-scalar-row (dest y-plane cb-plane cr-plane y img-w chroma-w)
  (declare (type (simple-array (unsigned-byte 8) (*)) dest y-plane cb-plane cr-plane)
           (type fixnum y img-w chroma-w)
           (optimize (speed 3) (safety 0)))
  (let* ((yc       (the fixnum (ash y -1)))
         (y-off    (the fixnum (* y img-w)))
         (c-off    (the fixnum (* yc chroma-w)))
         (dest-off (the fixnum (* y (the fixnum (* img-w 3))))))
    (declare (type fixnum yc y-off c-off dest-off))
    (loop for x fixnum from 0 below img-w
          for xc fixnum = (the fixnum (ash x -1))
          for dest-idx fixnum = (the fixnum (+ dest-off (the fixnum (* x 3))))
          for yy fixnum = (aref y-plane (the fixnum (+ y-off x)))
          for cb fixnum = (aref cb-plane (the fixnum (+ c-off xc)))
          for cr fixnum = (aref cr-plane (the fixnum (+ c-off xc))) do
          (setf (aref dest dest-idx)
                (the uint8 (limit (+ yy (aref +cb-b-tab+ cb)))))
          (setf (aref dest (+ dest-idx 1))
                (the uint8 (limit (+ yy (ash (+ (aref +cb-g-tab+ cb)
                                                (aref +cr-g-tab+ cr))
                                             (- shift))))))
          (setf (aref dest (+ dest-idx 2))
                (the uint8 (limit (+ yy (aref +cr-r-tab+ cr))))))))

#+(and sbcl x86-64)
(defun fused-420-colorspace-convert-simd (dest y-plane cb-plane cr-plane img-w img-h chroma-w)
  "Fused AVX-512BW + VBMI 4:2:0 color conversion: reads Y at full resolution (W x H)
and Cb/Cr at native quarter resolution (Wc x Hc). Duplicates chroma samples in-register
using VPERMB and calculates chroma deltas once per 2-row pair (y and y+1).

Pinning Invariant:
The raw SAP pointers (dest-sap, y-sap, cb-sap, cr-sap) are pinned for the entire duration
of the processing loop via WITH-PINNED-OBJECTS. No consing, memory allocation, formatting,
or error handling may occur inside this inner loop. This guarantees that the GC will not
relocate arrays or invalidate pointers during dereferences."
  (declare (type (simple-array (unsigned-byte 8) (*)) dest y-plane cb-plane cr-plane)
           (type fixnum img-w img-h chroma-w)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((nchunks32   (floor img-w 32))
         (simd-cols   (* nchunks32 32))
         (int-mask    (sb-simd-avx512bw:u8.64-aref +int-bgr-mask+ 0))
         (dup-mask    *fused-chroma-dup-mask*)
         (bias        (sb-simd-avx512bw:s16.32 128))
         (zero-s16    (sb-simd-avx512bw:s16.32 0))
         (max-s16     (sb-simd-avx512bw:s16.32 255))
         (zero-u8-32  (sb-simd-avx2:u8.32 0))
         (zero-u8-64  (sb-simd-avx512bw:u8.64 0))
         (k-cr-r      (sb-simd-avx512bw:s16.32 +ycc-cr-r+))
         (k-cb-g      (sb-simd-avx512bw:s16.32 +ycc-cb-g+))
         (k-cr-g      (sb-simd-avx512bw:s16.32 +ycc-cr-g+))
         (k-cb-b      (sb-simd-avx512bw:s16.32 +ycc-cb-b+))
         (npx         (* img-w img-h))
         (chroma-h   (the fixnum (ash (the fixnum (1+ img-h)) -1)))
         (chroma-npx (* chroma-w chroma-h)))
    (declare (type sb-simd-avx512bw:u8.64 int-mask dup-mask zero-u8-64)
             (type sb-simd-avx2:u8.32 zero-u8-32)
             (type sb-simd-avx512bw:s16.32 bias zero-s16 max-s16 k-cr-r k-cb-g k-cr-g k-cb-b)
             (type fixnum nchunks32 simd-cols npx chroma-h chroma-npx))

    ;; Non-elidable buffer bounds checks prior to SAP dereference
    (when (< (length dest) (the fixnum (* npx 3)))
      (error "Destination buffer too small: ~D < ~D" (length dest) (* npx 3)))
    (when (< (length y-plane) npx)
      (error "Y plane buffer too small: ~D < ~D" (length y-plane) npx))
    (when (< (length cb-plane) chroma-npx)
      (error "Cb plane buffer too small: ~D < ~D" (length cb-plane) chroma-npx))
    (when (< (length cr-plane) chroma-npx)
      (error "Cr plane buffer too small: ~D < ~D" (length cr-plane) chroma-npx))

    (sb-sys:with-pinned-objects (dest y-plane cb-plane cr-plane)
      (let ((dest-sap (sb-sys:vector-sap dest))
            (y-sap    (sb-sys:vector-sap y-plane))
            (cb-sap   (sb-sys:vector-sap cb-plane))
            (cr-sap   (sb-sys:vector-sap cr-plane)))
        (macrolet ((convert-row (y-row-off d-row-off)
                     `(let* ((y32      (sb-simd-avx2:u8.32-sap-ref y-sap (+ ,y-row-off x)))
                             (y16      (sb-simd-avx512bw:s16.32!
                                        (sb-simd-avx512bw:u16.32-from-u8.32 y32)))
                             (r        (sb-simd-avx512bw:s16.32+ y16 rd))
                             (g        (sb-simd-avx512bw:s16.32+ y16 gd))
                             (b        (sb-simd-avx512bw:s16.32+ y16 bd))
                             (r-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 r)))
                             (g-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 g)))
                             (b-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 b)))
                             (rb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! r-clamp)))
                             (gb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! g-clamp)))
                             (bb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! b-clamp)))
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
                             (mid16-1  (sb-simd-avx2:u8.16-from-u8.32 (sb-simd-avx512bw:u8.32-from-u8.64 out-1 1) 0))
                             (d-idx    (+ ,d-row-off (* chunk 96))))
                        (setf (sb-simd-avx2:u8.32-sap-ref dest-sap d-idx) low32-0)
                        (setf (sb-simd-sse2:u8.16-sap-ref dest-sap (+ d-idx 32)) mid16-0)
                        (setf (sb-simd-avx2:u8.32-sap-ref dest-sap (+ d-idx 48)) low32-1)
                        (setf (sb-simd-sse2:u8.16-sap-ref dest-sap (+ d-idx 80)) mid16-1))))
          (loop for y fixnum from 0 below img-h by 2 do
            (let* ((yc        (the fixnum (ash y -1)))
                   (y1        (the fixnum (1+ y)))
                   (has-y1    (< y1 img-h))
                   (y-off0    (the fixnum (* y img-w)))
                   (y-off1    (the fixnum (* y1 img-w)))
                   (c-off     (the fixnum (* yc chroma-w)))
                   (dest-off0 (the fixnum (* y (the fixnum (* img-w 3)))))
                   (dest-off1 (the fixnum (* y1 (the fixnum (* img-w 3))))))
              (declare (type fixnum yc y1 y-off0 y-off1 c-off dest-off0 dest-off1)
                       (type boolean has-y1))
              ;; 1. SIMD chunks across rows y and y1
              (loop for chunk fixnum from 0 below nchunks32
                    for x fixnum = (* chunk 32)
                    for xc fixnum = (* chunk 16) do
                      (let* ((cb16     (sb-simd-sse2:u8.16-sap-ref cb-sap (+ c-off xc)))
                             (cr16     (sb-simd-sse2:u8.16-sap-ref cr-sap (+ c-off xc)))
                             ;; Duplicate 16 chroma bytes to 32 bytes via VPERMB
                             (cb-in32  (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 cb16 0))
                             (cr-in32  (sb-simd-avx2:u8.32-insert-u8.16 zero-u8-32 cr16 0))
                             (cb-in64  (sb-simd-avx512bw:u8.64-insert-u8.32 zero-u8-64 cb-in32 0))
                             (cr-in64  (sb-simd-avx512bw:u8.64-insert-u8.32 zero-u8-64 cr-in32 0))
                             (cb-dup64 (sb-simd-avx512vbmi:u8.64-permute dup-mask cb-in64))
                             (cr-dup64 (sb-simd-avx512vbmi:u8.64-permute dup-mask cr-in64))
                             (cb32     (sb-simd-avx512bw:u8.32-from-u8.64 cb-dup64 0))
                             (cr32     (sb-simd-avx512bw:u8.32-from-u8.64 cr-dup64 0))
                             (cb16-s   (sb-simd-avx512bw:s16.32!
                                        (sb-simd-avx512bw:u16.32-from-u8.32 cb32)))
                             (cr16-s   (sb-simd-avx512bw:s16.32!
                                        (sb-simd-avx512bw:u16.32-from-u8.32 cr32)))
                             (cb-prime (sb-simd-avx512bw:s16.32- cb16-s bias))
                             (cr-prime (sb-simd-avx512bw:s16.32- cr16-s bias))
                             ;; Compute chroma deltas ONCE for both rows
                             (rd       (sb-simd-avx512bw:s16.32+
                                        cr-prime
                                        (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-r)))
                             (gd       (sb-simd-avx512bw:s16.32+
                                        (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-g)
                                        (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-g)))
                             (bd       (sb-simd-avx512bw:s16.32+
                                        cb-prime
                                        (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-b))))
                        ;; Convert row y
                        (convert-row y-off0 dest-off0)
                        ;; Convert row y1 if within height
                        (when has-y1
                          (convert-row y-off1 dest-off1))))

              ;; 2. Trailing scalar pixels
              (when (< simd-cols img-w)
                (loop for x fixnum from simd-cols below img-w
                      for xc fixnum = (the fixnum (ash x -1))
                      for d-idx0 fixnum = (+ dest-off0 (* x 3))
                      for d-idx1 fixnum = (+ dest-off1 (* x 3))
                      for yy0 fixnum = (aref y-plane (+ y-off0 x))
                      for cb fixnum = (aref cb-plane (+ c-off xc))
                      for cr fixnum = (aref cr-plane (+ c-off xc))
                      for b-val = (limit (+ yy0 (aref +cb-b-tab+ cb)))
                      for g-val = (limit (+ yy0 (ash (+ (aref +cb-g-tab+ cb)
                                                        (aref +cr-g-tab+ cr))
                                                     (- shift))))
                      for r-val = (limit (+ yy0 (aref +cr-r-tab+ cr))) do
                        (setf (aref dest d-idx0) b-val
                              (aref dest (+ d-idx0 1)) g-val
                              (aref dest (+ d-idx0 2)) r-val)
                        (when has-y1
                          (let ((yy1 (aref y-plane (+ y-off1 x))))
                            (setf (aref dest d-idx1) (limit (+ yy1 (aref +cb-b-tab+ cb)))
                                  (aref dest (+ d-idx1 1)) (limit (+ yy1 (ash (+ (aref +cb-g-tab+ cb)
                                                                                 (aref +cr-g-tab+ cr))
                                                                              (- shift))))
                                  (aref dest (+ d-idx1 2)) (limit (+ yy1 (aref +cr-r-tab+ cr)))))))))))))))

#+(and sbcl x86-64)
(defun planar-444-colorspace-convert-simd (dest y-plane cb-plane cr-plane npixels)
  "Convert contiguous 4:4:4 planar Y, Cb, Cr into interleaved BGR in DEST for NPIXELS.
Processes 32 pixels per loop iteration using AVX-512BW and pinned SAPs.
Returns number of pixels processed.

Pinning Invariant:
The raw SAP pointers (dest-sap, y-sap, cb-sap, cr-sap) are pinned for the entire duration
of the processing loop via WITH-PINNED-OBJECTS. No consing, memory allocation, formatting,
or error handling may occur inside this inner loop."
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

    ;; Non-elidable buffer bounds checks prior to SAP dereference
    (when (< (length dest) (the fixnum (* npixels 3)))
      (error "Destination buffer too small: ~D < ~D" (length dest) (* npixels 3)))
    (when (< (length y-plane) npixels)
      (error "Y plane buffer too small: ~D < ~D" (length y-plane) npixels))
    (when (< (length cb-plane) npixels)
      (error "Cb plane buffer too small: ~D < ~D" (length cb-plane) npixels))
    (when (< (length cr-plane) npixels)
      (error "Cr plane buffer too small: ~D < ~D" (length cr-plane) npixels))

    (sb-sys:with-pinned-objects (dest y-plane cb-plane cr-plane)
      (let ((dest-sap (sb-sys:vector-sap dest))
            (y-sap    (sb-sys:vector-sap y-plane))
            (cb-sap   (sb-sys:vector-sap cb-plane))
            (cr-sap   (sb-sys:vector-sap cr-plane)))
        (loop for chunk fixnum from 0 below nchunks32
              for px-off fixnum = (* chunk 32)
              for dest-off fixnum = (* chunk 96) do
                (let* ((y32      (sb-simd-avx2:u8.32-sap-ref y-sap px-off))
                       (cb32     (sb-simd-avx2:u8.32-sap-ref cb-sap px-off))
                       (cr32     (sb-simd-avx2:u8.32-sap-ref cr-sap px-off))
                       (y16      (sb-simd-avx512bw:s16.32!
                                  (sb-simd-avx512bw:u16.32-from-u8.32 y32)))
                       (cb16     (sb-simd-avx512bw:s16.32!
                                  (sb-simd-avx512bw:u16.32-from-u8.32 cb32)))
                       (cr16     (sb-simd-avx512bw:s16.32!
                                  (sb-simd-avx512bw:u16.32-from-u8.32 cr32)))
                       (cb-prime (sb-simd-avx512bw:s16.32- cb16 bias))
                       (cr-prime (sb-simd-avx512bw:s16.32- cr16 bias))
                       (rd       (sb-simd-avx512bw:s16.32+
                                  cr-prime
                                  (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-r)))
                       (gd       (sb-simd-avx512bw:s16.32+
                                  (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-g)
                                  (sb-simd-avx512bw:s16.32-mulhrs cr-prime k-cr-g)))
                       (bd       (sb-simd-avx512bw:s16.32+
                                  cb-prime
                                  (sb-simd-avx512bw:s16.32-mulhrs cb-prime k-cb-b)))
                       (r        (sb-simd-avx512bw:s16.32+ y16 rd))
                       (g        (sb-simd-avx512bw:s16.32+ y16 gd))
                       (b        (sb-simd-avx512bw:s16.32+ y16 bd))
                       (r-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 r)))
                       (g-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 g)))
                       (b-clamp  (sb-simd-avx512bw:s16.32-max zero-s16 (sb-simd-avx512bw:s16.32-min max-s16 b)))
                       (rb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! r-clamp)))
                       (gb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! g-clamp)))
                       (bb       (sb-simd-avx512bw:u8.32-from-u16.32 (sb-simd-avx512bw:u16.32! b-clamp)))
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
                  (setf (sb-simd-avx2:u8.32-sap-ref dest-sap dest-off) low32-0)
                  (setf (sb-simd-sse2:u8.16-sap-ref dest-sap (+ dest-off 32)) mid16-0)
                  (setf (sb-simd-avx2:u8.32-sap-ref dest-sap (+ dest-off 48)) low32-1)
                  (setf (sb-simd-sse2:u8.16-sap-ref dest-sap (+ dest-off 80)) mid16-1))))
      (* nchunks32 32))))

(defun planar-colorspace-convert-simd-accelerated (image)
  (declare (optimize (speed 3) (safety 0)))
  (let* ((dest     (descriptor-buffer image))
         (y-plane  (descriptor-y-plane image))
         (cb-plane (descriptor-cb-plane image))
         (cr-plane (descriptor-cr-plane image))
         (img-w    (descriptor-width image))
         (img-h    (descriptor-height image))
         (is-420   (= (the fixnum (aref (descriptor-H image) 0)) 2))
         (chroma-w (if is-420 (the fixnum (ash (the fixnum (1+ img-w)) -1)) img-w)))
    (declare (type (simple-array (unsigned-byte 8) (*)) dest y-plane cb-plane cr-plane)
             (type fixnum img-w img-h chroma-w)
             (type boolean is-420))
    #+(and sbcl x86-64)
    (if *simd-decode-color-p*
        (if is-420
            (fused-420-colorspace-convert-simd dest y-plane cb-plane cr-plane img-w img-h chroma-w)
            (let* ((npx (* img-w img-h))
                   (chunk-limit (floor npx 32))
                   (required-dest-len (+ (* chunk-limit 96) 64))
                   (safe-chunks (if (>= (length dest) required-dest-len)
                                    chunk-limit
                                    (max 0 (floor (- (length dest) 64) 96))))
                   (simd-pixels (* safe-chunks 32))
                   (done (if (> simd-pixels 0)
                             (planar-444-colorspace-convert-simd dest y-plane cb-plane cr-plane simd-pixels)
                             0)))
              (declare (type fixnum npx done))
              (when (< done npx)
                (planar-colorspace-convert-scalar-tail dest y-plane cb-plane cr-plane done npx))))
        (if is-420
            (dotimes (y img-h)
              (fused-420-colorspace-convert-scalar-row dest y-plane cb-plane cr-plane y img-w chroma-w))
            (planar-colorspace-convert-scalar-tail dest y-plane cb-plane cr-plane 0 (* img-w img-h))))
    #-(and sbcl x86-64)
    (if is-420
        (dotimes (y img-h)
          (fused-420-colorspace-convert-scalar-row dest y-plane cb-plane cr-plane y img-w chroma-w))
        (planar-colorspace-convert-scalar-tail dest y-plane cb-plane cr-plane 0 (* img-w img-h)))))

;;; Accelerated Entry Point for Decoder
;;;
;;; Drop-in replacement for INVERSE-COLORSPACE-CONVERT.

(defun inverse-colorspace-convert-simd-accelerated (image)
  "Drop-in replacement for INVERSE-COLORSPACE-CONVERT that dispatches to
planar SIMD kernel if descriptor has planar buffers enabled, else to
interleaved SIMD kernel, falling back to scalar for any trailing pixels."
  (declare (optimize (speed 3) (safety 0)))
  (if (descriptor-planar-p image)
      (planar-colorspace-convert-simd-accelerated image)
      (let* ((buffer (descriptor-buffer image))
             (width  (descriptor-width image))
             (height (descriptor-height image))
             (npx    (* width height)))
        (declare (type (simple-array (unsigned-byte 8) (*)) buffer)
                 (type fixnum width height npx))
        #+(and sbcl x86-64)
        (if *simd-decode-color-p*
            (let* ((chunk-limit (floor npx 16))
                   (required-len (+ (* chunk-limit 48) 16))
                   ;; When buffer has >= 64 bytes trailing padding (as allocated by
                   ;; ALLOCATE-BUFFER), safe-chunks = chunk-limit always and the
                   ;; tail only handles (< 16) trailing pixels.
                   (safe-chunks (if (>= (length buffer) required-len)
                                    chunk-limit
                                    (max 0 (floor (- (length buffer) 16) 48))))
                   (simd-pixels (* safe-chunks 16))
                   (done (if (> simd-pixels 0)
                             (inverse-colorspace-convert-simd buffer simd-pixels)
                             0)))
              (declare (type fixnum done))
              (when (< done npx)
                (inverse-colorspace-convert-scalar-tail buffer done npx)))
            (inverse-colorspace-convert-scalar-tail buffer 0 npx))
        #-(and sbcl x86-64)
        (inverse-colorspace-convert-scalar-tail buffer 0 npx))))

;;; Forward LLM-DCT Accelerated Kernel (Layout B: 8x8 Register Transpose)
;;;
;;; Target: AMD Zen 5 (AVX2).
;;; Correctness: Bit-for-bit identical to scalar CL-JPEG LLM-DCT across all test cases.
;;; Note on fixed-point constants: Constants deviate from libjpeg-turbo in the final bit
;;; due to a legacy rounding convention, preserved for compatibility with existing CL-JPEG output.
;;; Strategy: Each 8x8 block is loaded into 8 256-bit registers (one row per register,
;;; widened from s16 to s32). Transposed into columns via 24 unpack/permute instructions,
;;; computed via 1D vertical DCT (Pass 1), transposed back, computed via 1D vertical DCT
;;; (Pass 2), and packed back into s16.16 pairs for storage.

#+(and sbcl x86-64)
(defmacro transpose-8x8 (r0 r1 r2 r3 r4 r5 r6 r7)
             `(let* ((s0 (sb-simd-avx2:s32.8-unpacklo ,r0 ,r1))
                     (s1 (sb-simd-avx2:s32.8-unpackhi ,r0 ,r1))
                     (s2 (sb-simd-avx2:s32.8-unpacklo ,r2 ,r3))
                     (s3 (sb-simd-avx2:s32.8-unpackhi ,r2 ,r3))
                     (s4 (sb-simd-avx2:s32.8-unpacklo ,r4 ,r5))
                     (s5 (sb-simd-avx2:s32.8-unpackhi ,r4 ,r5))
                     (s6 (sb-simd-avx2:s32.8-unpacklo ,r6 ,r7))
                     (s7 (sb-simd-avx2:s32.8-unpackhi ,r6 ,r7))
                     (t0 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpacklo (sb-simd-avx2:s64.4! s0) (sb-simd-avx2:s64.4! s2))))
                     (t1 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpackhi (sb-simd-avx2:s64.4! s0) (sb-simd-avx2:s64.4! s2))))
                     (t2 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpacklo (sb-simd-avx2:s64.4! s1) (sb-simd-avx2:s64.4! s3))))
                     (t3 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpackhi (sb-simd-avx2:s64.4! s1) (sb-simd-avx2:s64.4! s3))))
                     (t4 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpacklo (sb-simd-avx2:s64.4! s4) (sb-simd-avx2:s64.4! s6))))
                     (t5 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpackhi (sb-simd-avx2:s64.4! s4) (sb-simd-avx2:s64.4! s6))))
                     (t6 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpacklo (sb-simd-avx2:s64.4! s5) (sb-simd-avx2:s64.4! s7))))
                     (t7 (sb-simd-avx2:s32.8! (sb-simd-avx2:s64.4-unpackhi (sb-simd-avx2:s64.4! s5) (sb-simd-avx2:s64.4! s7)))))
                (values
                 (sb-simd-avx2:s32.8-permute128 t0 t4 #x20)
                 (sb-simd-avx2:s32.8-permute128 t1 t5 #x20)
                 (sb-simd-avx2:s32.8-permute128 t2 t6 #x20)
                 (sb-simd-avx2:s32.8-permute128 t3 t7 #x20)
                 (sb-simd-avx2:s32.8-permute128 t0 t4 #x31)
                 (sb-simd-avx2:s32.8-permute128 t1 t5 #x31)
                 (sb-simd-avx2:s32.8-permute128 t2 t6 #x31)
                 (sb-simd-avx2:s32.8-permute128 t3 t7 #x31))))
#+(and sbcl x86-64)
(macrolet ((vert-dct-p1 (c0 c1 c2 c3 c4 c5 c6 c7)
             `(let* ((tmp0 (sb-simd-avx2:s32.8+ ,c0 ,c7))
                     (tmp7 (sb-simd-avx2:s32.8- ,c0 ,c7))
                     (tmp1 (sb-simd-avx2:s32.8+ ,c1 ,c6))
                     (tmp6 (sb-simd-avx2:s32.8- ,c1 ,c6))
                     (tmp2 (sb-simd-avx2:s32.8+ ,c2 ,c5))
                     (tmp5 (sb-simd-avx2:s32.8- ,c2 ,c5))
                     (tmp3 (sb-simd-avx2:s32.8+ ,c3 ,c4))
                     (tmp4 (sb-simd-avx2:s32.8- ,c3 ,c4))
                     (tmp10 (sb-simd-avx2:s32.8+ tmp0 tmp3))
                     (tmp13 (sb-simd-avx2:s32.8- tmp0 tmp3))
                     (tmp11 (sb-simd-avx2:s32.8+ tmp1 tmp2))
                     (tmp12 (sb-simd-avx2:s32.8- tmp1 tmp2))
                     (out0 (sb-simd-avx2:s32.8-shiftl (sb-simd-avx2:s32.8+ tmp10 tmp11) 1))
                     (out4 (sb-simd-avx2:s32.8-shiftl (sb-simd-avx2:s32.8- tmp10 tmp11) 1))
                     (z1 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ tmp12 tmp13) (sb-simd-avx2:s32.8 ,+FIX-0-541196100+)))
                     (out2 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo tmp13 (sb-simd-avx2:s32.8 ,+FIX-0-765366865+)) (sb-simd-avx2:s32.8 2048)) 12))
                     (out6 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo tmp12 (sb-simd-avx2:s32.8 ,(- +FIX-1-847759065+))) (sb-simd-avx2:s32.8 2048)) 12))
                     (z1-odd (sb-simd-avx2:s32.8+ tmp4 tmp7))
                     (z2 (sb-simd-avx2:s32.8+ tmp5 tmp6))
                     (z3 (sb-simd-avx2:s32.8+ tmp4 tmp6))
                     (z4 (sb-simd-avx2:s32.8+ tmp5 tmp7))
                     (z5 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ z3 z4) (sb-simd-avx2:s32.8 ,+FIX-1-175875602+)))
                     (m4 (sb-simd-avx2:s32.8-mullo tmp4 (sb-simd-avx2:s32.8 ,+FIX-0-298631336+)))
                     (m5 (sb-simd-avx2:s32.8-mullo tmp5 (sb-simd-avx2:s32.8 ,+FIX-2-053119869+)))
                     (m6 (sb-simd-avx2:s32.8-mullo tmp6 (sb-simd-avx2:s32.8 ,+FIX-3-072711026+)))
                     (m7 (sb-simd-avx2:s32.8-mullo tmp7 (sb-simd-avx2:s32.8 ,+FIX-1-501321110+)))
                     (z1-odd-m (sb-simd-avx2:s32.8-mullo z1-odd (sb-simd-avx2:s32.8 ,(- +FIX-0-899976223+))))
                     (z2-m (sb-simd-avx2:s32.8-mullo z2 (sb-simd-avx2:s32.8 ,(- +FIX-2-562915447+))))
                     (z3-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z3 (sb-simd-avx2:s32.8 ,(- +FIX-1-961570560+))) z5))
                     (z4-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z4 (sb-simd-avx2:s32.8 ,(- +FIX-0-390180644+))) z5))
                     (bias (sb-simd-avx2:s32.8 2048))
                     (out7 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m4 z1-odd-m z3-m bias) 12))
                     (out5 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m5 z2-m z4-m bias) 12))
                     (out3 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m6 z2-m z3-m bias) 12))
                     (out1 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m7 z1-odd-m z4-m bias) 12)))
                (values out0 out1 out2 out3 out4 out5 out6 out7)))
           (vert-dct-p2 (c0 c1 c2 c3 c4 c5 c6 c7)
             `(let* ((tmp0 (sb-simd-avx2:s32.8+ ,c0 ,c7))
                     (tmp7 (sb-simd-avx2:s32.8- ,c0 ,c7))
                     (tmp1 (sb-simd-avx2:s32.8+ ,c1 ,c6))
                     (tmp6 (sb-simd-avx2:s32.8- ,c1 ,c6))
                     (tmp2 (sb-simd-avx2:s32.8+ ,c2 ,c5))
                     (tmp5 (sb-simd-avx2:s32.8- ,c2 ,c5))
                     (tmp3 (sb-simd-avx2:s32.8+ ,c3 ,c4))
                     (tmp4 (sb-simd-avx2:s32.8- ,c3 ,c4))
                     (tmp10 (sb-simd-avx2:s32.8+ tmp0 tmp3))
                     (tmp13 (sb-simd-avx2:s32.8- tmp0 tmp3))
                     (tmp11 (sb-simd-avx2:s32.8+ tmp1 tmp2))
                     (tmp12 (sb-simd-avx2:s32.8- tmp1 tmp2))
                     (out0 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ tmp10 tmp11 (sb-simd-avx2:s32.8 1)) 1))
                     (out4 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8- tmp10 tmp11) (sb-simd-avx2:s32.8 1)) 1))
                     (z1 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ tmp12 tmp13) (sb-simd-avx2:s32.8 ,+FIX-0-541196100+)))
                     (out2 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo tmp13 (sb-simd-avx2:s32.8 ,+FIX-0-765366865+)) (sb-simd-avx2:s32.8 8192)) 14))
                     (out6 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo tmp12 (sb-simd-avx2:s32.8 ,(- +FIX-1-847759065+))) (sb-simd-avx2:s32.8 8192)) 14))
                     (z1-odd (sb-simd-avx2:s32.8+ tmp4 tmp7))
                     (z2 (sb-simd-avx2:s32.8+ tmp5 tmp6))
                     (z3 (sb-simd-avx2:s32.8+ tmp4 tmp6))
                     (z4 (sb-simd-avx2:s32.8+ tmp5 tmp7))
                     (z5 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ z3 z4) (sb-simd-avx2:s32.8 ,+FIX-1-175875602+)))
                     (m4 (sb-simd-avx2:s32.8-mullo tmp4 (sb-simd-avx2:s32.8 ,+FIX-0-298631336+)))
                     (m5 (sb-simd-avx2:s32.8-mullo tmp5 (sb-simd-avx2:s32.8 ,+FIX-2-053119869+)))
                     (m6 (sb-simd-avx2:s32.8-mullo tmp6 (sb-simd-avx2:s32.8 ,+FIX-3-072711026+)))
                     (m7 (sb-simd-avx2:s32.8-mullo tmp7 (sb-simd-avx2:s32.8 ,+FIX-1-501321110+)))
                     (z1-odd-m (sb-simd-avx2:s32.8-mullo z1-odd (sb-simd-avx2:s32.8 ,(- +FIX-0-899976223+))))
                     (z2-m (sb-simd-avx2:s32.8-mullo z2 (sb-simd-avx2:s32.8 ,(- +FIX-2-562915447+))))
                     (z3-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z3 (sb-simd-avx2:s32.8 ,(- +FIX-1-961570560+))) z5))
                     (z4-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z4 (sb-simd-avx2:s32.8 ,(- +FIX-0-390180644+))) z5))
                     (bias (sb-simd-avx2:s32.8 8192))
                     (out7 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m4 z1-odd-m z3-m bias) 14))
                     (out5 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m5 z2-m z4-m bias) 14))
                     (out3 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m6 z2-m z3-m bias) 14))
                     (out1 (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ m7 z1-odd-m z4-m bias) 14)))
                (values out0 out1 out2 out3 out4 out5 out6 out7))))

  (defun llm-dct-simd (buf b nblocks)
    "Accelerated 8x8 2D forward DCT for CL-JPEG Strategy A planar layout.
Operates on block B (0-indexed) within BUF containing NBLOCKS blocks.
Output is bit-for-bit identical to scalar LLM-DCT."
    (declare (optimize (speed 3) (safety 0) (debug 0))
             (type (simple-array sint16 (*)) buf)
             (type fixnum b nblocks))
    (let* ((stride (the fixnum (ash nblocks 3)))
           (b_base (the fixnum (ash b 3)))
           ;; 1. Load rows into s32.8
           (r0 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 0 stride)))))
           (r1 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 1 stride)))))
           (r2 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 2 stride)))))
           (r3 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 3 stride)))))
           (r4 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 4 stride)))))
           (r5 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 5 stride)))))
           (r6 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 6 stride)))))
           (r7 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref buf (+ b_base (* 7 stride))))))
      ;; 2. Transpose so rows become columns
      (multiple-value-bind (t0 t1 t2 t3 t4 t5 t6 t7)
          (transpose-8x8 r0 r1 r2 r3 r4 r5 r6 r7)
        ;; 3. Pass 1: 1D DCT on rows (now vertical in t0..t7)
        (multiple-value-bind (p0 p1 p2 p3 p4 p5 p6 p7)
            (vert-dct-p1 t0 t1 t2 t3 t4 t5 t6 t7)
          ;; 4. Transpose so columns become rows
          (multiple-value-bind (c0 c1 c2 c3 c4 c5 c6 c7)
              (transpose-8x8 p0 p1 p2 p3 p4 p5 p6 p7)
            ;; 5. Pass 2: 1D DCT on columns (vertical in c0..c7)
            (multiple-value-bind (f0 f1 f2 f3 f4 f5 f6 f7)
                (vert-dct-p2 c0 c1 c2 c3 c4 c5 c6 c7)
              ;; 6. Pack pairs of rows and store
              (let* ((p01 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f0 f1)) #xD8)))
                     (p23 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f2 f3)) #xD8)))
                     (p45 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f4 f5)) #xD8)))
                     (p67 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f6 f7)) #xD8))))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 0 stride))) (sb-simd-avx2:s16.8-from-s16.16 p01 0))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 1 stride))) (sb-simd-avx2:s16.8-from-s16.16 p01 1))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 2 stride))) (sb-simd-avx2:s16.8-from-s16.16 p23 0))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 3 stride))) (sb-simd-avx2:s16.8-from-s16.16 p23 1))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 4 stride))) (sb-simd-avx2:s16.8-from-s16.16 p45 0))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 5 stride))) (sb-simd-avx2:s16.8-from-s16.16 p45 1))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 6 stride))) (sb-simd-avx2:s16.8-from-s16.16 p67 0))
                (setf (sb-simd-avx2:s16.8-aref buf (+ b_base (* 7 stride))) (sb-simd-avx2:s16.8-from-s16.16 p67 1))
                (values))))))))

;;; Inverse LLM-DCT Accelerated Kernel
;;;
;;; Target: AMD Zen 5 (AVX2).
;;; Correctness: Bit-for-bit identical to scalar CL-JPEG INVERSE-LLM-DCT across all test cases.
;;; Strategy: Load 8 rows into s32.8 registers, multiply by Q-table rows.
;;; Pass 1: 1D IDCT on columns (vert-idct-p1), wrapping to signed 16-bit to match scalar ws.
;;; Transpose via transpose-8x8.
;;; Pass 2: 1D IDCT on rows (vert-idct-p2) with modular clamping matching *idct-limit-array*.
;;; Transpose back to row-major order via transpose-8x8.
;;; Pack pairs of rows into s16.16 registers and store into block.

#+(and sbcl x86-64)
(progn
  (declaim (inline load-q-row-s32))
  (defun load-q-row-s32 (q-row)
    (declare (type (simple-array uint8 (8)) q-row)
             (optimize (speed 3) (safety 0)))
    (sb-simd-avx2:make-s32.8 (aref q-row 0)
                             (aref q-row 1)
                             (aref q-row 2)
                             (aref q-row 3)
                             (aref q-row 4)
                             (aref q-row 5)
                             (aref q-row 6)
                             (aref q-row 7)))

  (defmacro vert-idct-p1 (c0 c1 c2 c3 c4 c5 c6 c7)
    `(let* ((z2 ,c2)
            (z3 ,c6)
            (z1 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ z2 z3) (sb-simd-avx2:s32.8 ,+FIX-0-541196100+)))
            (tmp2 (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo z3 (sb-simd-avx2:s32.8 ,(- +FIX-1-847759065+)))))
            (tmp3 (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo z2 (sb-simd-avx2:s32.8 ,+FIX-0-765366865+))))
            (z2-0 ,c0)
            (z3-4 ,c4)
            (tmp0 (sb-simd-avx2:s32.8-shiftl (sb-simd-avx2:s32.8+ z2-0 z3-4) 13))
            (tmp1 (sb-simd-avx2:s32.8-shiftl (sb-simd-avx2:s32.8- z2-0 z3-4) 13))
            (tmp10 (sb-simd-avx2:s32.8+ tmp0 tmp3))
            (tmp13 (sb-simd-avx2:s32.8- tmp0 tmp3))
            (tmp11 (sb-simd-avx2:s32.8+ tmp1 tmp2))
            (tmp12 (sb-simd-avx2:s32.8- tmp1 tmp2))
            (in7 ,c7)
            (in5 ,c5)
            (in3 ,c3)
            (in1 ,c1)
            (z1-odd (sb-simd-avx2:s32.8+ in7 in1))
            (z2-odd (sb-simd-avx2:s32.8+ in5 in3))
            (z3-odd (sb-simd-avx2:s32.8+ in7 in3))
            (z4-odd (sb-simd-avx2:s32.8+ in5 in1))
            (z5 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ z3-odd z4-odd) (sb-simd-avx2:s32.8 ,+FIX-1-175875602+)))
            (m0 (sb-simd-avx2:s32.8-mullo in7 (sb-simd-avx2:s32.8 ,+FIX-0-298631336+)))
            (m1 (sb-simd-avx2:s32.8-mullo in5 (sb-simd-avx2:s32.8 ,+FIX-2-053119869+)))
            (m2 (sb-simd-avx2:s32.8-mullo in3 (sb-simd-avx2:s32.8 ,+FIX-3-072711026+)))
            (m3 (sb-simd-avx2:s32.8-mullo in1 (sb-simd-avx2:s32.8 ,+FIX-1-501321110+)))
            (z1-m (sb-simd-avx2:s32.8-mullo z1-odd (sb-simd-avx2:s32.8 ,(- +FIX-0-899976223+))))
            (z2-m (sb-simd-avx2:s32.8-mullo z2-odd (sb-simd-avx2:s32.8 ,(- +FIX-2-562915447+))))
            (z3-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z3-odd (sb-simd-avx2:s32.8 ,(- +FIX-1-961570560+))) z5))
            (z4-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z4-odd (sb-simd-avx2:s32.8 ,(- +FIX-0-390180644+))) z5))
            (ot0 (sb-simd-avx2:s32.8+ m0 z1-m z3-m))
            (ot1 (sb-simd-avx2:s32.8+ m1 z2-m z4-m))
            (ot2 (sb-simd-avx2:s32.8+ m2 z2-m z3-m))
            (ot3 (sb-simd-avx2:s32.8+ m3 z1-m z4-m))
            (bias (sb-simd-avx2:s32.8 2048)))
       (macrolet ((descale-wrap16 (val)
                    `(let ((d (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ ,val bias) 12)))
                       (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8-shiftl d 16) 16))))
         (values
          (descale-wrap16 (sb-simd-avx2:s32.8+ tmp10 ot3))
          (descale-wrap16 (sb-simd-avx2:s32.8+ tmp11 ot2))
          (descale-wrap16 (sb-simd-avx2:s32.8+ tmp12 ot1))
          (descale-wrap16 (sb-simd-avx2:s32.8+ tmp13 ot0))
          (descale-wrap16 (sb-simd-avx2:s32.8- tmp13 ot0))
          (descale-wrap16 (sb-simd-avx2:s32.8- tmp12 ot1))
          (descale-wrap16 (sb-simd-avx2:s32.8- tmp11 ot2))
          (descale-wrap16 (sb-simd-avx2:s32.8- tmp10 ot3))))))

  (defmacro vert-idct-p2 (c0 c1 c2 c3 c4 c5 c6 c7)
    `(let* ((z2 ,c2)
            (z3 ,c6)
            (z1 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ z2 z3) (sb-simd-avx2:s32.8 ,+FIX-0-541196100+)))
            (tmp2 (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo z3 (sb-simd-avx2:s32.8 ,(- +FIX-1-847759065+)))))
            (tmp3 (sb-simd-avx2:s32.8+ z1 (sb-simd-avx2:s32.8-mullo z2 (sb-simd-avx2:s32.8 ,+FIX-0-765366865+))))
            (z2-0 ,c0)
            (z3-4 ,c4)
            (tmp0 (sb-simd-avx2:s32.8-shiftl (sb-simd-avx2:s32.8+ z2-0 z3-4) 13))
            (tmp1 (sb-simd-avx2:s32.8-shiftl (sb-simd-avx2:s32.8- z2-0 z3-4) 13))
            (tmp10 (sb-simd-avx2:s32.8+ tmp0 tmp3))
            (tmp13 (sb-simd-avx2:s32.8- tmp0 tmp3))
            (tmp11 (sb-simd-avx2:s32.8+ tmp1 tmp2))
            (tmp12 (sb-simd-avx2:s32.8- tmp1 tmp2))
            (in7 ,c7)
            (in5 ,c5)
            (in3 ,c3)
            (in1 ,c1)
            (z1-odd (sb-simd-avx2:s32.8+ in7 in1))
            (z2-odd (sb-simd-avx2:s32.8+ in5 in3))
            (z3-odd (sb-simd-avx2:s32.8+ in7 in3))
            (z4-odd (sb-simd-avx2:s32.8+ in5 in1))
            (z5 (sb-simd-avx2:s32.8-mullo (sb-simd-avx2:s32.8+ z3-odd z4-odd) (sb-simd-avx2:s32.8 ,+FIX-1-175875602+)))
            (m0 (sb-simd-avx2:s32.8-mullo in7 (sb-simd-avx2:s32.8 ,+FIX-0-298631336+)))
            (m1 (sb-simd-avx2:s32.8-mullo in5 (sb-simd-avx2:s32.8 ,+FIX-2-053119869+)))
            (m2 (sb-simd-avx2:s32.8-mullo in3 (sb-simd-avx2:s32.8 ,+FIX-3-072711026+)))
            (m3 (sb-simd-avx2:s32.8-mullo in1 (sb-simd-avx2:s32.8 ,+FIX-1-501321110+)))
            (z1-m (sb-simd-avx2:s32.8-mullo z1-odd (sb-simd-avx2:s32.8 ,(- +FIX-0-899976223+))))
            (z2-m (sb-simd-avx2:s32.8-mullo z2-odd (sb-simd-avx2:s32.8 ,(- +FIX-2-562915447+))))
            (z3-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z3-odd (sb-simd-avx2:s32.8 ,(- +FIX-1-961570560+))) z5))
            (z4-m (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8-mullo z4-odd (sb-simd-avx2:s32.8 ,(- +FIX-0-390180644+))) z5))
            (ot0 (sb-simd-avx2:s32.8+ m0 z1-m z3-m))
            (ot1 (sb-simd-avx2:s32.8+ m1 z2-m z4-m))
            (ot2 (sb-simd-avx2:s32.8+ m2 z2-m z3-m))
            (ot3 (sb-simd-avx2:s32.8+ m3 z1-m z4-m))
            (bias (sb-simd-avx2:s32.8 65536))
            (c255 (sb-simd-avx2:s32.8 255))
            (c511 (sb-simd-avx2:s32.8 511))
            (c128 (sb-simd-avx2:s32.8 128))
            (zero (sb-simd-avx2:s32.8 0)))
       (macrolet ((clamp-limit (val)
                    `(let ((idx (sb-simd-avx2:s32.8-and (sb-simd-avx2:s32.8+ ,val c255) c511)))
                       (sb-simd-avx2:s32.8-min
                        (sb-simd-avx2:s32.8-max
                         (sb-simd-avx2:s32.8- idx c128)
                         zero)
                        c255))))
         (values
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ tmp10 ot3 bias) 17))
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ tmp11 ot2 bias) 17))
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ tmp12 ot1 bias) 17))
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ tmp13 ot0 bias) 17))
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8- tmp13 ot0) bias) 17))
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8- tmp12 ot1) bias) 17))
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8- tmp11 ot2) bias) 17))
          (clamp-limit (sb-simd-avx2:s32.8-shiftr (sb-simd-avx2:s32.8+ (sb-simd-avx2:s32.8- tmp10 ot3) bias) 17))))))

  (defun inverse-llm-dct-simd (block q-table ws)
    "Accelerated 8x8 2D inverse DCT and dequantization for CL-JPEG decoder.
Operates on BLOCK (sint16-2d-array) with Q-TABLE (uint8-2d-array). WS is ignored.
Output is bit-for-bit identical to scalar INVERSE-LLM-DCT across all inputs."
    (declare (optimize (speed 3) (safety 0) (debug 0))
             (type sint16-2d-array block)
             (type uint8-2d-array q-table)
             (ignore ws))
    (let* ((b0 (aref block 0)) (b1 (aref block 1))
           (b2 (aref block 2)) (b3 (aref block 3))
           (b4 (aref block 4)) (b5 (aref block 5))
           (b6 (aref block 6)) (b7 (aref block 7))
           (r0 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b0 0)))
           (r1 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b1 0)))
           (r2 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b2 0)))
           (r3 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b3 0)))
           (r4 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b4 0)))
           (r5 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b5 0)))
           (r6 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b6 0)))
           (r7 (sb-simd-avx2:s32.8-from-s16.8 (sb-simd-avx2:s16.8-aref b7 0)))
           (d0 (sb-simd-avx2:s32.8-mullo r0 (load-q-row-s32 (aref q-table 0))))
           (d1 (sb-simd-avx2:s32.8-mullo r1 (load-q-row-s32 (aref q-table 1))))
           (d2 (sb-simd-avx2:s32.8-mullo r2 (load-q-row-s32 (aref q-table 2))))
           (d3 (sb-simd-avx2:s32.8-mullo r3 (load-q-row-s32 (aref q-table 3))))
           (d4 (sb-simd-avx2:s32.8-mullo r4 (load-q-row-s32 (aref q-table 4))))
           (d5 (sb-simd-avx2:s32.8-mullo r5 (load-q-row-s32 (aref q-table 5))))
           (d6 (sb-simd-avx2:s32.8-mullo r6 (load-q-row-s32 (aref q-table 6))))
           (d7 (sb-simd-avx2:s32.8-mullo r7 (load-q-row-s32 (aref q-table 7)))))
      (multiple-value-bind (w0 w1 w2 w3 w4 w5 w6 w7)
          (vert-idct-p1 d0 d1 d2 d3 d4 d5 d6 d7)
        (multiple-value-bind (t0 t1 t2 t3 t4 t5 t6 t7)
            (transpose-8x8 w0 w1 w2 w3 w4 w5 w6 w7)
          (multiple-value-bind (o0 o1 o2 o3 o4 o5 o6 o7)
              (vert-idct-p2 t0 t1 t2 t3 t4 t5 t6 t7)
            (multiple-value-bind (f0 f1 f2 f3 f4 f5 f6 f7)
                (transpose-8x8 o0 o1 o2 o3 o4 o5 o6 o7)
              (let* ((p01 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f0 f1)) #xD8)))
                     (p23 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f2 f3)) #xD8)))
                     (p45 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f4 f5)) #xD8)))
                     (p67 (sb-simd-avx2:s16.16! (sb-simd-avx2:f64.4-permute4x64 (sb-simd-avx2:f64.4! (sb-simd-avx2:s16.16-packs f6 f7)) #xD8))))
                (setf (sb-simd-avx2:s16.8-aref b0 0) (sb-simd-avx2:s16.8-from-s16.16 p01 0))
                (setf (sb-simd-avx2:s16.8-aref b1 0) (sb-simd-avx2:s16.8-from-s16.16 p01 1))
                (setf (sb-simd-avx2:s16.8-aref b2 0) (sb-simd-avx2:s16.8-from-s16.16 p23 0))
                (setf (sb-simd-avx2:s16.8-aref b3 0) (sb-simd-avx2:s16.8-from-s16.16 p23 1))
                (setf (sb-simd-avx2:s16.8-aref b4 0) (sb-simd-avx2:s16.8-from-s16.16 p45 0))
                (setf (sb-simd-avx2:s16.8-aref b5 0) (sb-simd-avx2:s16.8-from-s16.16 p45 1))
                (setf (sb-simd-avx2:s16.8-aref b6 0) (sb-simd-avx2:s16.8-from-s16.16 p67 0))
                (setf (sb-simd-avx2:s16.8-aref b7 0) (sb-simd-avx2:s16.8-from-s16.16 p67 1))
                (values))))))))))

;;; Hook Registration

(eval-when (:load-toplevel :execute)
  (when *simd-decode-color-p*
    (setf *inverse-colorspace-convert-impl*
          #'inverse-colorspace-convert-simd-accelerated))
  (when *simd-encode-dct-p*
    (setf *llm-dct-impl*
          #'llm-dct-simd))
  (when *simd-decode-dct-p*
    (setf *inverse-llm-dct-impl*
          #'inverse-llm-dct-simd))
  (when *simd-planar-upsample-p*
    (setf *upsample-impl*
          #'planar-upsample)))

