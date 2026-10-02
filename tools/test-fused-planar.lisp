(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

;;; Define chroma duplication mask for AVX-512VBMI permute
(defvar *fused-chroma-dup-mask*
  (let ((arr (make-array 64 :element-type '(unsigned-byte 8)
                            :initial-contents (loop for i from 0 below 64 collect (if (< i 32) (floor i 2) 0)))))
    #+(and sbcl x86-64)
    (sb-simd-avx512bw:u8.64-aref arr 0)
    #-(and sbcl x86-64)
    nil))

(defun fused-upsample-420 (image scan block blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)
  "Fused planar upsampling: stores Y at full resolution (W x H) and Cb/Cr at native
reduced chroma resolution (W/2 x H/2). No 2x2 chroma sample replication."
  (declare (optimize (speed 3) (safety 0))
           (type sint16-2d-array block)
           (type fixnum blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend))
  (let* ((xbase-luma (the fixnum (+ (the fixnum (scan-x scan)) blk-x)))
         (ybase-luma (the fixnum (+ (the fixnum (scan-y scan)) blk-y)))
         (img-w (descriptor-width image))
         (img-h (descriptor-height image))
         (chroma-w (the fixnum (ash (the fixnum (1+ img-w)) -1)))
         (chroma-h (the fixnum (ash (the fixnum (1+ img-h)) -1))))
    (declare (type fixnum xbase-luma ybase-luma img-w img-h chroma-w chroma-h))
    (cond
      ;; 1. Luma (comp-index = 0)
      ((= comp-index 0)
       (let ((plane (descriptor-y-plane image)))
         (unless plane
           (return-from fused-upsample-420
             (upsample-default image scan block blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)))
         (if (and (<= (+ xbase-luma 8) img-w) (<= (+ ybase-luma 8) img-h))
             (upsample-1x1-planar-fast plane block (+ (* ybase-luma img-w) xbase-luma) img-w)
             (when (and (< xbase-luma img-w) (< ybase-luma img-h))
               (let ((max-r (min 8 (- img-h ybase-luma)))
                     (max-c (min 8 (- img-w xbase-luma))))
                 (declare (type fixnum max-r max-c))
                 (dotimes (r max-r)
                   (let ((row (aref block r))
                         (pos (+ (* (+ ybase-luma r) img-w) xbase-luma)))
                     (declare (type sint16-array row) (type fixnum pos))
                     (dotimes (c max-c)
                       (setf (aref plane (+ pos c)) (aref row c))))))))))

      ;; 2. Chroma Cb (comp-index = 1) or Cr (comp-index = 2) at native resolution
      ((or (= comp-index 1) (= comp-index 2))
       (let ((plane (if (= comp-index 1) (descriptor-cb-plane image) (descriptor-cr-plane image)))
             ;; Convert MCU luma position to native chroma coordinate
             (xbase-c (the fixnum (ash xbase-luma -1)))
             (ybase-c (the fixnum (ash ybase-luma -1))))
         (declare (type fixnum xbase-c ybase-c))
         (unless plane
           (return-from fused-upsample-420
             (upsample-default image scan block blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)))
         (if (and (<= (+ xbase-c 8) chroma-w) (<= (+ ybase-c 8) chroma-h))
             (upsample-1x1-planar-fast plane block (+ (* ybase-c chroma-w) xbase-c) chroma-w)
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

      ;; Fallback
      (t
       (upsample-default image scan block blk-x blk-y samp-h samp-v comp-index nwidth nw nx dend)))))

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
  "Fused AVX-512BW color conversion: reads Y at full resolution and Cb/Cr at native
quarter-resolution, duplicating chroma samples in-register via AVX-512VBMI VPERMB."
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

    ;; Pinning Invariant Check:
    ;; Verify that arrays satisfy required lengths before dereferencing SAPs.
    (assert (>= (length dest) (* npx 3)))
    (assert (>= (length y-plane) npx))
    (assert (>= (length cb-plane) chroma-npx))
    (assert (>= (length cr-plane) chroma-npx))

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

(defun fused-colorspace-convert (image)
  "Entry point for fused planar color conversion."
  (declare (optimize (speed 3) (safety 0)))
  (let* ((dest     (descriptor-buffer image))
         (y-plane  (descriptor-y-plane image))
         (cb-plane (descriptor-cb-plane image))
         (cr-plane (descriptor-cr-plane image))
         (img-w    (descriptor-width image))
         (img-h    (descriptor-height image))
         (chroma-w (the fixnum (ash (the fixnum (1+ img-w)) -1))))
    (declare (type (simple-array (unsigned-byte 8) (*)) dest y-plane cb-plane cr-plane)
             (type fixnum img-w img-h chroma-w))
    #+(and sbcl x86-64)
    (if *simd-decode-color-p*
        (fused-420-colorspace-convert-simd dest y-plane cb-plane cr-plane img-w img-h chroma-w)
        (dotimes (y img-h)
          (fused-420-colorspace-convert-scalar-row dest y-plane cb-plane cr-plane y img-w chroma-w)))
    #-(and sbcl x86-64)
    (dotimes (y img-h)
      (fused-420-colorspace-convert-scalar-row dest y-plane cb-plane cr-plane y img-w chroma-w))))

(defun decode-with-fused (path &optional desc)
  (let ((*upsample-impl* #'fused-upsample-420)
        (*inverse-colorspace-convert-impl* #'fused-colorspace-convert)
        (*use-simd* :on))
    (decode-image path :descriptor desc)))

(defun test-fused-suite ()
  (format t "~%====================================================================~%")
  (format t "  CL-JPEG FUSED PLANAR SIMD VALIDATION & PERFORMANCE SUITE~%")
  (format t "====================================================================~%")

  (dolist (item (list (list "/tmp/photo1_512x512.jpg" "photo1 512x512")
                      (list "/tmp/photo3_1024x1024.jpg" "photo3 1024x1024")
                      (list "/tmp/cl-jpeg-bench-512x512.jpg" "gradient 512x512")
                      (list "/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg" "testorig (odd size 227x149)")))
    (let ((path (first item))
          (name (second item)))
      (when (probe-file path)
        (format t "~%--- Testing ~A ---~%" name)
        ;; 1. Reference interleaved SIMD decode
        (let* ((ref-buf (let ((*use-simd* :on) (*upsample-impl* nil))
                          (decode-image path)))
               ;; 2. Fused planar decode
               (fused-buf (decode-with-fused path))
               (mismatches 0)
               (max-diff 0)
               (n (length ref-buf)))
          (dotimes (i n)
            (let ((diff (abs (- (aref ref-buf i) (aref fused-buf i)))))
              (when (> diff 0)
                (incf mismatches)
                (when (> diff max-diff)
                  (setf max-diff diff)))))
          (if (= mismatches 0)
              (format t "  [BIT-EXACT] 0 / ~D mismatches, max-diff=0 (100% BIT-EXACT MATCH!)~%" n)
              (format t "  [PARITY] ~D / ~D mismatches (~,2F%), max-diff=~D~%"
                      mismatches n (* (/ mismatches (float n)) 100.0) max-diff))))))

  (format t "~%====================================================================~%")
  (format t "  BENCHMARK: Interleaved vs Full-Planar vs Fused-Planar (50 reps)~%")
  (format t "====================================================================~%")
  (let ((path "/tmp/photo3_1024x1024.jpg"))
    ;; Warmup
    (dotimes (i 5)
      (let ((*use-simd* :on) (*upsample-impl* nil)) (decode-image path))
      (decode-with-fused path))

    (format t "1. Interleaved SIMD (single-shot):~%")
    (sb-ext:gc :full t)
    (time (dotimes (i 50)
            (let ((*use-simd* :on) (*upsample-impl* nil))
              (decode-image path))))

    (format t "~%2. Fused Planar SIMD (single-shot, native chroma planes):~%")
    (sb-ext:gc :full t)
    (time (dotimes (i 50)
            (decode-with-fused path))))

  (format t "~%====================================================================~%")
  (format t "  BENCHMARK: Descriptor Reuse (50 reps)~%")
  (format t "====================================================================~%")
  (let ((path "/tmp/photo3_1024x1024.jpg")
        (desc-int (make-descriptor))
        (desc-fus (make-descriptor)))
    (format t "1. Interleaved SIMD (reused descriptor):~%")
    (sb-ext:gc :full t)
    (time (dotimes (i 50)
            (let ((*use-simd* :on) (*upsample-impl* nil))
              (decode-image path :descriptor desc-int))))

    (format t "~%2. Fused Planar SIMD (reused descriptor):~%")
    (sb-ext:gc :full t)
    (time (dotimes (i 50)
            (decode-with-fused path desc-fus)))))

(test-fused-suite)

