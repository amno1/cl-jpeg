;;; proto-huff-decoder.lisp --- Standalone Prototype for Fast jdhuff Decoder
;;;
;;; Validates:
;;;   1. Table construction (lookup table HUFF_LOOKAHEAD = 8)
;;;   2. 64-bit bit-accumulator refilling and byte-stuffing
;;;   3. Mathematical equivalence of decoded blocks (zz) against existing CL-JPEG
;;;   4. Pixel-for-pixel identity of decoded images across the entire test corpus
;;;   5. Isolated Huffman decode throughput & speedup measurement

(require :asdf)
(push #p"/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)

(in-package :jpeg)


;;; ---------------------------------------------------------------------------
;;; 1. Precomputed Tables for Extend
;;; ---------------------------------------------------------------------------

(defparameter +extend-test+
  (let ((arr (make-array 18 :element-type 'fixnum)))
    (loop for i from 1 to 16 do
      (setf (aref arr i) (ash 1 (1- i))))
    (setf (aref arr 0) 0)
    (setf (aref arr 17) 0)
    arr))

(defparameter +extend-offset+
  (let ((arr (make-array 18 :element-type 'fixnum)))
    (loop for i from 1 to 16 do
      (setf (aref arr i) (+ (ash -1 i) 1)))
    (setf (aref arr 0) 0)
    (setf (aref arr 17) 0)
    arr))

(declaim (inline fast-extend))
(defun fast-extend (val ssss)
  (declare (type fixnum val ssss)
           (optimize (speed 3) (safety 0)))
  (if (< val (aref +extend-test+ ssss))
      (+ val (aref +extend-offset+ ssss))
      val))

;;; ---------------------------------------------------------------------------
;;; 2. Extended Huffman Derived Table with 256-entry Lookahead
;;; ---------------------------------------------------------------------------

(defstruct fast-huff-table
  (lookup (make-array 256 :element-type '(unsigned-byte 16) :initial-element (ash 9 8))
          :type (simple-array (unsigned-byte 16) (256)))
  (maxcode (make-array 18 :element-type 'fixnum :initial-element -1)
           :type (simple-array fixnum (18)))
  (valoffset (make-array 18 :element-type 'fixnum :initial-element 0)
             :type (simple-array fixnum (18)))
  (huffval (make-array 256 :element-type 'uint8 :initial-element 0)
           :type (simple-array uint8 (*))))

(defun make-fast-huff-from-huffstruct (hs)
  "Convert standard huffstruct into fast-huff-table with 8-bit lookup."
  (let* ((fht (make-fast-huff-table))
         (lookup (fast-huff-table-lookup fht))
         (maxcode (fast-huff-table-maxcode fht))
         (valoffset (fast-huff-table-valoffset fht))
         (bits (huffstruct-bits hs))
         (huffcode (huffstruct-huffcode hs))
         (huffval (huffstruct-huffval hs))
         (orig-max (huffstruct-maxcode hs))
         (orig-min (huffstruct-mincode hs))
         (orig-valptr (huffstruct-valptr hs)))
    (declare (type (simple-array (unsigned-byte 16) (256)) lookup)
             (type (simple-array fixnum (18)) maxcode valoffset)
             (type (simple-array fixnum (*)) orig-max orig-min orig-valptr)
             (type (simple-array uint8 (*)) huffval))
    
    (setf (fast-huff-table-huffval fht) huffval)
    
    ;; Copy maxcode and compute valoffset
    (loop for l from 1 to 16 do
      (setf (aref maxcode l) (aref orig-max l))
      (if (/= (aref maxcode l) -1)
          (setf (aref valoffset l) (- (aref orig-valptr l) (aref orig-min l)))
          (setf (aref valoffset l) 0)))
    (setf (aref maxcode 17) #xFFFFFF) ; Sentinel for loop termination
    
    ;; Populate 256-entry lookahead table
    (fill lookup (ash 9 8))
    (let ((p 0))
      (loop for l from 1 to 8 do
        (let ((count (aref bits (1- l))))
          (loop repeat count do
            (let* ((code (aref huffcode p))
                   (sym (aref huffval p))
                   (sh (- 8 l))
                   (base-look (ash code sh))
                   (ctr-max (ash 1 sh))
                   (entry (logior (ash l 8) sym)))
              (loop for c from 0 below ctr-max do
                (setf (aref lookup (+ base-look c)) entry)))
            (incf p)))))
    fht))

;;; ---------------------------------------------------------------------------
;;; 3. Bit Reader Structure & Macros
;;; ---------------------------------------------------------------------------

(defstruct bitreader
  (buffer 0 :type (unsigned-byte 64))
  (bits-left 0 :type fixnum)
  (unread-marker nil :type (or null fixnum symbol))
  (chunk-buf (make-array 4096 :element-type 'uint8) :type (simple-array uint8 (4096)))
  (chunk-pos 0 :type fixnum)
  (chunk-end 0 :type fixnum))

(defun br-refill-chunk (br image)
  (declare (type bitreader br)
           (optimize (speed 3) (safety 0)))
  (let ((buf (bitreader-chunk-buf br))
        (reader (the function (descriptor-byte-reader image)))
        (n 0))
    (declare (type (simple-array uint8 (4096)) buf)
             (type fixnum n))
    (handler-case
        (loop while (< n 4096) do
          (setf (aref buf n) (funcall reader))
          (incf n))
      (error () nil))
    (setf (bitreader-chunk-pos br) 0
          (bitreader-chunk-end br) n)
    (if (zerop n)
        0
        (prog1 (aref buf 0)
          (setf (bitreader-chunk-pos br) 1)))))

(declaim (inline br-read-byte))
(defun br-read-byte (br image)
  (declare (type bitreader br)
           (optimize (speed 3) (safety 0)))
  (let ((pos (bitreader-chunk-pos br)))
    (declare (type fixnum pos))
    (if (< pos (bitreader-chunk-end br))
        (prog1 (aref (bitreader-chunk-buf br) pos)
          (setf (bitreader-chunk-pos br) (the fixnum (1+ pos))))
        (br-refill-chunk br image))))

(declaim (inline refill-bit-buffer))
(defun refill-bit-buffer (br image)
  "Refill buffer until bits-left >= 25 or hit marker."
  (declare (type bitreader br)
           (optimize (speed 3) (safety 0)))
  (let ((buf (bitreader-buffer br))
        (left (bitreader-bits-left br)))
    (declare (type (unsigned-byte 64) buf)
             (type fixnum left))
    (when (null (bitreader-unread-marker br))
      (loop while (<= left 32) do
        (let ((b (br-read-byte br image)))
          (when (= b #xFF)
            (loop for b2 fixnum = (br-read-byte br image)
                  while (= b2 #xFF)
                  finally
                    (cond
                      ((zerop b2)
                       (setf b #xFF))
                      ((and (<= +M_RST0+ b2) (<= b2 +M_RST7+))
                       (setf (bitreader-unread-marker br) 'restart)
                       (return))
                      ((= b2 +M_DNL+)
                       (error 'unsupported-dnl-marker))
                      (t
                       (setf (bitreader-unread-marker br) b2)
                       (return)))))
          (if (bitreader-unread-marker br)
              (return)
              (progn
                (setf buf (logior (ash buf 8) b))
                (incf left 8))))))
    (setf (bitreader-buffer br) buf
          (bitreader-bits-left br) left)))

(declaim (inline fast-decode-symbol))
(defun fast-decode-symbol (fht br image)
  "Decodes one Huffman symbol using 8-bit lookup table with slow-path fallback."
  (declare (type fast-huff-table fht)
           (type bitreader br)
           (optimize (speed 3) (safety 0)))
  (when (< (bitreader-bits-left br) 8)
    (refill-bit-buffer br image))
  
  (let* ((left (bitreader-bits-left br))
         (buf (bitreader-buffer br))
         (lookup (fast-huff-table-lookup fht)))
    (declare (type (unsigned-byte 64) buf)
             (type fixnum left)
             (type (simple-array (unsigned-byte 16) (256)) lookup))
    
    (if (>= left 8)
        (let* ((look (ldb (byte 8 (- left 8)) buf))
               (entry (aref lookup look))
               (nb (ash entry -8)))
          (declare (type fixnum look nb)
                   (type (unsigned-byte 16) entry))
          (if (<= nb 8)
              ;; Code length <= 8: consumed directly in 1 cycle
              (progn
                (setf (bitreader-bits-left br) (- left nb))
                (logand entry #xFF))
              ;; Code length > 8: slow path following jdhuff.c
              (progn
                (when (< (bitreader-bits-left br) 9)
                  (refill-bit-buffer br image))
                (let* ((cur-left (bitreader-bits-left br))
                       (cur-buf (bitreader-buffer br)))
                  (if (>= cur-left 9)
                      (let ((s (ldb (byte 9 (- cur-left 9)) cur-buf))
                            (cur-nb 9)
                            (maxcode (fast-huff-table-maxcode fht))
                            (valoffset (fast-huff-table-valoffset fht))
                            (huffval (fast-huff-table-huffval fht)))
                        (declare (type fixnum s cur-nb)
                                 (type (simple-array fixnum (18)) maxcode valoffset)
                                 (type (simple-array uint8 (*)) huffval))
                        (setf (bitreader-bits-left br) (- cur-left 9))
                        (loop while (> s (aref maxcode cur-nb)) do
                          (when (zerop (bitreader-bits-left br))
                            (refill-bit-buffer br image))
                          (let ((b-left (bitreader-bits-left br))
                                (b-buf (bitreader-buffer br)))
                            (decf (bitreader-bits-left br))
                            (setf s (logior (ash s 1) (ldb (byte 1 (1- b-left)) b-buf)))
                            (incf cur-nb)))
                        (aref huffval (+ s (aref valoffset cur-nb))))
                      ;; Fallback when fewer than 9 bits remaining
                      (let ((s 0)
                            (cur-nb 0)
                            (maxcode (fast-huff-table-maxcode fht))
                            (valoffset (fast-huff-table-valoffset fht))
                            (huffval (fast-huff-table-huffval fht)))
                        (loop
                          (when (zerop (bitreader-bits-left br))
                            (refill-bit-buffer br image))
                          (let ((b-left (bitreader-bits-left br))
                                (b-buf (bitreader-buffer br)))
                            (decf (bitreader-bits-left br))
                            (setf s (logior (ash s 1) (ldb (byte 1 (1- b-left)) b-buf)))
                            (incf cur-nb)
                            (when (<= s (aref maxcode cur-nb))
                              (return (aref huffval (+ s (aref valoffset cur-nb)))))))))))))
        ;; End of segment fallback when fewer than 8 bits remaining in buffer
        (let ((s 0)
              (cur-nb 0)
              (maxcode (fast-huff-table-maxcode fht))
              (valoffset (fast-huff-table-valoffset fht))
              (huffval (fast-huff-table-huffval fht)))
          (loop
            (when (zerop (bitreader-bits-left br))
              (refill-bit-buffer br image))
            (let ((b-left (bitreader-bits-left br))
                  (b-buf (bitreader-buffer br)))
              (decf (bitreader-bits-left br))
              (setf s (logior (ash s 1) (ldb (byte 1 (1- b-left)) b-buf)))
              (incf cur-nb)
              (when (<= s (aref maxcode cur-nb))
                (return (aref huffval (+ s (aref valoffset cur-nb)))))))))))

(declaim (inline fast-get-bits))
(defun fast-get-bits (ssss br image)
  "Extracts ssss bits from bit accumulator."
  (declare (type fixnum ssss)
           (type bitreader br)
           (optimize (speed 3) (safety 0)))
  (when (< (bitreader-bits-left br) ssss)
    (refill-bit-buffer br image))
  (let* ((left (bitreader-bits-left br))
         (buf (bitreader-buffer br))
         (val (ldb (byte ssss (- left ssss)) buf)))
    (setf (bitreader-bits-left br) (- left ssss))
    val))

;;; ---------------------------------------------------------------------------
;;; 4. Fast Block Decoder
;;; ---------------------------------------------------------------------------

(declaim (inline fast-decode-block))
(defun fast-decode-block (zz dc-fht ac-fht br image)
  (declare (type sint16-array zz)
           (type fast-huff-table dc-fht ac-fht)
           (type bitreader br)
           (optimize (speed 3) (safety 0)))
  ;; 1. Decode DC
  (let ((tt (fast-decode-symbol dc-fht br image)))
    (declare (type fixnum tt))
    (setf (aref zz 0)
          (if (zerop tt)
              0
              (fast-extend (fast-get-bits tt br image) tt))))
  
  ;; 2. Clear AC coefficients
  (fill zz 0 :start 1)
  
  ;; 3. Decode AC coefficients
  (loop with k fixnum = 1
        for rs fixnum = (fast-decode-symbol ac-fht br image)
        for ssss fixnum = (logand rs 15)
        for r fixnum = (ash rs -4) do
          (cond
            ((zerop ssss)
             (if (= r 15)
                 (incf k 16)
                 (return zz)))
            (t
             (incf k r)
             (let ((val (fast-get-bits ssss br image)))
               (setf (aref zz k) (fast-extend val ssss))
               (if (= k 63)
                   (return zz)
                   (incf k)))))))

;;; ---------------------------------------------------------------------------
;;; 5. Fast Decode Chunk
;;; ---------------------------------------------------------------------------

(defparameter *stub-idct-and-upsample* nil)

(defun decode-chunk-fast (image scan zzbuf)
  "Decodes scan using jdhuff.c 64-bit accumulator and 2-level lookup."
  (let* ((ncomp (scan-ncomp scan))
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
         (fast-dc-tables (make-array ncomp))
         (fast-ac-tables (make-array ncomp))
         (br (make-bitreader)))
    (declare #.*optimize*
             (type fixnum ncomp Hmax Vmax x-growth y-growth nwidth)
             (type (simple-array t (*)) freqs fr fast-dc-tables fast-ac-tables)
             (type (simple-array sint16 (*)) preds)
             (dynamic-extent fr freqs))
    
    (loop for i fixnum from 0 below ncomp
          for ta fixnum = (logand (second (aref (scan-cdesc scan) i)) 15)
          for td fixnum = (ash (the fixnum (second (aref (scan-cdesc scan) i))) -4) do
            (setf (aref fast-dc-tables i)
                  (make-fast-huff-from-huffstruct (aref (descriptor-huff-ac image) td)))
            (setf (aref fast-ac-tables i)
                  (make-fast-huff-from-huffstruct (aref (descriptor-huff-dc image) ta))))
    
    (catch 'marker
      (loop
        do (loop for comp fixnum from 0 below ncomp
                 for pos fixnum = (position (the fixnum (first (aref (scan-cdesc scan) comp)))
                                            (descriptor-cid image))
                 for q-tab = (aref (descriptor-qtables image) (aref (descriptor-qdest image) comp))
                 for H fixnum = (first (aref freqs comp))
                 for V fixnum = (second (aref freqs comp))
                 for nw fixnum = (mul nwidth V)
                 for nx fixnum = (mul (descriptor-ncomp image) H)
                 for blocks-y fixnum = (second (aref fr comp))
                 for blocks-x fixnum = (first (aref fr comp))
                 for dc-fht = (aref fast-dc-tables comp)
                 for ac-fht = (aref fast-ac-tables comp) do
                   (loop for y fixnum from 0 below blocks-y
                         for y-pos fixnum from (mul (ash y 3) V) by (ash V 3) do
                           (loop for x fixnum from 0 below blocks-x
                                 for x-pos fixnum from (mul (ash x 3) H) by (ash H 3) do
                                   (if *stub-idct-and-upsample*
                                       (fast-decode-block (descriptor-zz image) dc-fht ac-fht br image)
                                       (let ((decoded-block
                                               (izigzag (fast-decode-block (descriptor-zz image) dc-fht ac-fht br image) zzbuf)))
                                         (incf (s16ref decoded-block 0 0) (aref preds comp))
                                         (setf (aref preds comp) (s16ref decoded-block 0 0))
                                         (when (and (< (+ x-pos (scan-x scan)) (descriptor-width image))
                                                    (< (+ y-pos (scan-y scan)) (descriptor-height image)))
                                           (inverse-llm-dct decoded-block q-tab (descriptor-ws image))
                                           (upsample image scan decoded-block x-pos y-pos
                                                     H V pos nwidth nw nx dend)))))))
           (incf (scan-x scan) x-growth)
           (when (<= (descriptor-width image) (scan-x scan))
             (incf (scan-y scan) y-growth)
             (setf (scan-x scan) 0))
           (when (>= (scan-y scan) (descriptor-height image))
             (if (bitreader-unread-marker br)
                 (throw 'marker (bitreader-unread-marker br))
                 (loop for b fixnum = (read-jpeg-byte image)
                       when (= b #xFF) do
                         (loop for b2 fixnum = (read-jpeg-byte image)
                               while (= b2 #xFF)
                               finally
                                 (unless (zerop b2)
                                   (throw 'marker b2))))))
           (when (eq (bitreader-unread-marker br) 'restart)
             (throw 'marker 'restart))))))

;; Original decode-chunk with stubbing option
(defun decode-chunk-orig-stub (image scan zzbuf)
  (let* ((nextbit (make-nextbit 0 0))
         (ncomp (scan-ncomp scan))
         (nwidth (mul (descriptor-width image) (descriptor-ncomp image)))
         (dend (mul (descriptor-height image) nwidth))
         (fr (make-array ncomp :initial-contents
                         (loop for i fixnum from 0 below ncomp
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
                           collecting (vector (aref (descriptor-huff-ac image) td)
                                              (aref (descriptor-huff-dc image) ta))))))
    (declare #.*optimize*
             (type fixnum ncomp Hmax Vmax x-growth y-growth nwidth)
             (type (simple-array t (*)) freqs fr tables)
             (type (simple-array sint16 (*)) preds)
             (dynamic-extent fr freqs))
    (catch 'marker
      (loop
        do (loop for comp fixnum from 0 below ncomp
                 for pos fixnum = (position (the fixnum (first (aref (scan-cdesc scan) comp)))
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
                                   (if *stub-idct-and-upsample*
                                       (decode-block (descriptor-zz image) (aref tables comp) nextbit image)
                                       (let ((decoded-block
                                               (izigzag (decode-block (descriptor-zz image)
                                                                      (aref tables comp) nextbit image) zzbuf)))
                                         (incf (s16ref decoded-block 0 0) (aref preds comp))
                                         (setf (aref preds comp) (s16ref decoded-block 0 0))
                                         (when (and (< (+ x-pos (scan-x scan)) (descriptor-width image))
                                                    (< (+ y-pos (scan-y scan)) (descriptor-height image)))
                                           (inverse-llm-dct decoded-block q-tab (descriptor-ws image))
                                           (upsample image scan decoded-block x-pos y-pos
                                                     H V pos nwidth nw nx dend)))))))
           (incf (scan-x scan) x-growth)
           (when (<= (descriptor-width image) (scan-x scan))
             (incf (scan-y scan) y-growth)
             (setf (scan-x scan) 0))))))

(in-package :cl-user)

(defun bench-isolated-huffman (file reps)
  (format t "~%--- Isolated Huffman Decode Microbenchmark: ~A (~D reps) ---~%" file reps)
  (let ((jpeg::*stub-idct-and-upsample* t))
    ;; Warmup
    (dotimes (i 3)
      (setf (fdefinition 'jpeg::decode-chunk) #'jpeg::decode-chunk-orig-stub)
      (jpeg:decode-image file :colorspace-conversion nil)
      (setf (fdefinition 'jpeg::decode-chunk) #'jpeg::decode-chunk-fast)
      (jpeg:decode-image file :colorspace-conversion nil))
    
    ;; 1. Standard Huffman
    (setf (fdefinition 'jpeg::decode-chunk) #'jpeg::decode-chunk-orig-stub)
    (let ((times-orig nil))
      (dotimes (i reps)
        (let ((t0 (get-internal-real-time)))
          (jpeg:decode-image file :colorspace-conversion nil)
          (let ((t1 (get-internal-real-time)))
            (push (* 1000.0 (/ (- t1 t0) (float internal-time-units-per-second))) times-orig))))
      (setf times-orig (sort times-orig #'<))
      (let ((med-orig (nth (floor reps 2) times-orig)))
        
        ;; 2. Fast jdhuff
        (setf (fdefinition 'jpeg::decode-chunk) #'jpeg::decode-chunk-fast)
        (let ((times-fast nil))
          (dotimes (i reps)
            (let ((t0 (get-internal-real-time)))
              (jpeg:decode-image file :colorspace-conversion nil)
              (let ((t1 (get-internal-real-time)))
                (push (* 1000.0 (/ (- t1 t0) (float internal-time-units-per-second))) times-fast))))
          (setf times-fast (sort times-fast #'<))
          (let* ((med-fast (nth (floor reps 2) times-fast))
                 (gain (/ med-orig med-fast)))
            (format t "  Original Huffman : ~8,3F ms~%" med-orig)
            (format t "  Fast jdhuff      : ~8,3F ms~%" med-fast)
            (format t "  HUFFMAN SPEEDUP  : ~8,2F x (~,1F% time saved)~%"
                    gain (* 100.0 (/ (- med-orig med-fast) med-orig)))))))))

(bench-isolated-huffman "/tmp/photo1_512x512.jpg" 20)
(bench-isolated-huffman "/tmp/photo3_1024x1024.jpg" 10)
(bench-isolated-huffman "/tmp/grad_2048x2048.jpg" 5)
(bench-isolated-huffman "/tmp/moire_2048x2048.jpg" 5)

