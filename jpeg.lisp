;;; ANSI Common Lisp (mostly) baseline JPEG encoder/decoder implementation
;;; Copyright [c] 1999,2015-2017 Eugene Zaikonnikov <eugene@funcall.org>
;;;               
;;; This software is distributed under the terms of BSD-like license
;;; [see LICENSE for details]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; This software was sponsored by Kelly E. Murray
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Two main functions available:
;;;
;;; (encode-image filename image ncomp h w &key sampling q-tabs q-factor), where:
;;; filename - output file name
;;; ncomp - number of components (1-4)
;;; h, w - source image height and width respectively
;;; image - array of B, G, R pixels in case of three component image,
;;;         array of grayscale pixels in case of single component,
;;;         array of 2 or 4 pixels in the case of two or four component image respectively
;;; :q-tabs - specifies quantization tables vector, should be 1 for 1,
;;;           2 for 2, 2 for 3 and 4 entries for 4 components
;;; :sampling - sampling frequency for ncomp components by X and Y axis,
;;;             e.g. '((2 2) (1 1) (1 1)) for three components, can be omitted
;;;             for grayscale and RGB images
;;; :q-factor — quantization strength, 1 (finest) to 255 (coarsest), default 64
;;; Returns nothing of practical use
;;;
;;; (decode-image filename &key buffer (colorspace-conversion t))
;;; filename - jpeg file name
;;; Returns (multiple-valued) IMAGE array in the same format as encoder source image,
;;; image HEIGHT and image WIDTH
;;; A pre-allocated BUFFER can be specified (see JPEG:ALLOCATE-BUFFER).
;;; If :colorspace-conversion is NIL, no conversion from YUV space is performed.
;;;
;;; For those impatient additional function defined:
;;; (jpeg-to-bmp &key infile outfile)
;;; Converts JPEG image specified by infile into Microsoft Windows 24-bit BMP format (outfile),
;;; returns NIL
;;;
;;; Additionaly, you may use more user-friendly version of encode-image: encode-wrapper.
;;; (encoding-wrapper filename image ncomp h w &key quality)
;;; All parameters have the same meaning as in encode-image, except quality.
;;; It is an integer value ranging 1 to 5 which specifies
;;; subjective quality of a resulting image.

;;; Technical details: encoder produces interleaved jpeg file, without restarts.
;;; In a case of 3 components image will be written in JFIF format.

;;; Decoder can deal with *almost* all baseline jpeg files, regardless JFIF or not.

;;; It supports restarts, interleaved/noninterleaved files, multiscan images, 1 to 4 color
;;; channels, up to 4 quantization tables and two sets of huffman tables with random order
;;; of their definition inside the image. Decoder *does not* support DNL marker, due to
;;; it's rarity and amount of work needed to implement it, so decoder isn't baseline in a
;;; strict sense.

;;; Both encoder and decoder utilize Loeffer, Ligtenberg and Moschytz integer discrete
;;; cosine transform algorithms with 12 multiplications in each loop.

;;; Based on CCITT Rec. T.81
;;; "Information technology - digital compression and coding of continious-tone still images
;;; - requirements and guidelines".
;;; Credits:
;;; to the Independent JPEG Group -
;;; colorspace conversion and DCT algorithms were adopted from their sources;
;;; to Jeff Dalton for his wise paper "Common Lisp Pitfalls".

(in-package #:jpeg)

(declaim (inline csize quantize get-average zigzag read-jpeg-byte
                 llm-dct descale crunch colorspace-convert subsample inverse-llm-dct
                 dequantize upsample extend recieve decode-ac decode-dc decode-block
                 izigzag write-bits limit))

(deftype uint8 () '(unsigned-byte 8))
(deftype uint8-array () '(simple-array uint8 (*)))
(deftype uint8-2d-array () '(simple-array uint8-array (*)))

(deftype suint8 () '(signed-byte 8))
(deftype sint8-array () '(simple-array sint8 (*)))
(deftype sint8-2d-array () '(simple-array sint8-array (*)))

(deftype sint16 () '(signed-byte 16))
(deftype sint16-array () '(simple-array sint16 (*)))
(deftype sint16-2d-array () '(simple-array sint16-array (*)))

(deftype uint16 () '(unsigned-byte 16))
(deftype uint16-array () '(simple-array uint16 (*)))
(deftype uint16-2d-array () '(simple-array uint16-array (*)))

(deftype fixnum-array () '(simple-array fixnum (*)))
(deftype fixnum-2d-array () '(simple-array fixnum-array (*)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *optimize* '(optimize (safety 0) (space 0) (debug 0) (speed 3))))

;;; SIMD Acceleration Controls & Dispatch Hooks
(defparameter *use-simd* :auto
  "Controls SIMD acceleration. One of :auto (use if available), :on (force), :off (disable).")

(declaim (type (or null function) *inverse-colorspace-convert-impl*))
(defvar *inverse-colorspace-convert-impl* nil
  "Hook for SIMD-accelerated inverse colorspace conversion (decoder).")

(declaim (type (or null function) *llm-dct-impl*))
(defvar *llm-dct-impl* nil
  "Hook for SIMD-accelerated forward DCT (encoder).")

(declaim (type (or null function) *inverse-llm-dct-impl*))
(defvar *inverse-llm-dct-impl* nil
  "Hook for SIMD-accelerated inverse DCT (decoder).")

(declaim (type (or null function) *upsample-impl*))
(defvar *upsample-impl* nil
  "Hook for SIMD/planar-accelerated upsampling (decoder).")


(defparameter *chroma-subsampling* :box
  "Controls chroma downsampling method during encoding: :box (default, 2x2 box filter averaging for quality parity with cjpeg) or :point (legacy nearest-neighbor).")

(eval-when (:compile-toplevel :load-toplevel :execute)
;;; For ease of reference
(defmacro u8ref (data x y)
  `(the uint8 (aref (the uint8-array (aref (the uint8-2d-array ,data) ,y)) ,x)))

(defmacro s16ref (data x y)
  `(the sint16 (aref (the sint16-array (aref (the sint16-2d-array ,data) ,y)) ,x)))

(defmacro u16ref (data x y)
  `(the uint16 (aref (the uint16-array (aref (the uint16-2d-array ,data) ,y)) ,x)))

(defmacro fixref (data x y)
  `(the fixnum (aref (the fixnum-array (aref (the fixnum-2d-array ,data) ,y)) ,x)))

;;; Integer arithmetic wrappers
(defmacro plus (a b)
  `(the fixnum (+ (the fixnum ,a) (the fixnum ,b))))

(defmacro minus (a b)
  #+(or clisp abcl)
  `(- ,a ,b)
  #-(or clisp abcl)
  `(the fixnum (- (the fixnum ,a) (the fixnum ,b))))

(defmacro mul (a b)
  `(the fixnum (* (the fixnum ,a) (the fixnum ,b))))

(defmacro plus3 (x y z)
  `(plus (plus ,x ,y) ,z))

(defmacro mul3 (x y z)
  `(mul (mul ,x ,y) ,z)))

;;; Somewhat silly, but who knows...
(when (/= (integer-length most-positive-fixnum)
          (integer-length most-negative-fixnum))
  (error "Can't compile with asymmetric fixnums!"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Here we define some constants (markers, quantization and huffman tables etc.)

(defmacro define-constant (name value &optional doc)
  `(defconstant ,name (if (boundp ',name) (symbol-value ',name) ,value)
     ,@(when doc (list doc))))

(eval-when (:compile-toplevel :load-toplevel)

(defun uint8-array (&rest contents)
  (the uint8-array (make-array (length contents) :element-type 'uint8
			       :initial-contents contents)))
  
(defun 2d-uint8-array (&rest contents)
  (let ((nrow (length contents)))
    (the uint8-2d-array
	 (make-array nrow
		:element-type 'uint8-array
		:initial-contents
                (loop for row in contents
                      collecting (make-array (length row) :element-type 'uint8
                                             :initial-contents row))))))

(defun 2d-sint16-array (&rest contents)
  (let ((nrow (length contents)))
    (the sint16-2d-array
	 (make-array nrow
		:element-type 'sint16-array
		:initial-contents
                (loop for row in contents
                      collecting (make-array (length row) :element-type 'sint16
                                             :initial-contents row))))))

;;; Source huffman tables for the encoder
(define-constant +luminance-dc-bits+
    (uint8-array #x00 #x01 #x05 #x01 #x01 #x01 #x01 #x01
		 #x01 #x00 #x00 #x00 #x00 #x00 #x00 #x00))

(define-constant +luminance-dc-values+
    (uint8-array #x00 #x01 #x02 #x03 #x04 #x05 #x06 #x07 #x08 #x09 #x0a #x0b))

(define-constant +chrominance-dc-bits+
    (uint8-array #x00 #x03 #x01 #x01 #x01 #x01 #x01 #x01
		 #x01 #x01 #x01 #x00 #x00 #x00 #x00 #x00))

(define-constant +chrominance-dc-values+
    (uint8-array #x00 #x01 #x02 #x03 #x04 #x05 #x06 #x07 #x08 #x09 #x0a #x0b))

(define-constant +luminance-ac-bits+
    (uint8-array #x00 #x02 #x01 #x03 #x03 #x02 #x04 #x03
		 #x05 #x05 #x04 #x04 #x00 #x00 #x01 #x7d))

(define-constant +luminance-ac-values+
    (uint8-array
     #x01 #x02 #x03 #x00 #x04 #x11 #x05 #x12
     #x21 #x31 #x41 #x06 #x13 #x51 #x61 #x07
     #x22 #x71 #x14 #x32 #x81 #x91 #xa1 #x08
     #x23 #x42 #xb1 #xc1 #x15 #x52 #xd1 #xf0
     #x24 #x33 #x62 #x72 #x82 #x09 #x0a #x16
     #x17 #x18 #x19 #x1a #x25 #x26 #x27 #x28
     #x29 #x2a #x34 #x35 #x36 #x37 #x38 #x39
     #x3a #x43 #x44 #x45 #x46 #x47 #x48 #x49
     #x4a #x53 #x54 #x55 #x56 #x57 #x58 #x59
     #x5a #x63 #x64 #x65 #x66 #x67 #x68 #x69
     #x6a #x73 #x74 #x75 #x76 #x77 #x78 #x79
     #x7a #x83 #x84 #x85 #x86 #x87 #x88 #x89
     #x8a #x92 #x93 #x94 #x95 #x96 #x97 #x98
     #x99 #x9a #xa2 #xa3 #xa4 #xa5 #xa6 #xa7
     #xa8 #xa9 #xaa #xb2 #xb3 #xb4 #xb5 #xb6
     #xb7 #xb8 #xb9 #xba #xc2 #xc3 #xc4 #xc5
     #xc6 #xc7 #xc8 #xc9 #xca #xd2 #xd3 #xd4
     #xd5 #xd6 #xd7 #xd8 #xd9 #xda #xe1 #xe2
     #xe3 #xe4 #xe5 #xe6 #xe7 #xe8 #xe9 #xea
     #xf1 #xf2 #xf3 #xf4 #xf5 #xf6 #xf7 #xf8
     #xf9 #xfa))

(define-constant +chrominance-ac-bits+
    (uint8-array #x00 #x02 #x01 #x02 #x04 #x04 #x03 #x04
		#x07 #x05 #x04 #x04 #x00 #x01 #x02 #x77))

(define-constant +chrominance-ac-values+
    (uint8-array
     #x00 #x01 #x02 #x03 #x11 #x04 #x05 #x21
     #x31 #x06 #x12 #x41 #x51 #x07 #x61 #x71
     #x13 #x22 #x32 #x81 #x08 #x14 #x42 #x91
     #xa1 #xb1 #xc1 #x09 #x23 #x33 #x52 #xf0
     #x15 #x62 #x72 #xd1 #x0a #x16 #x24 #x34
     #xe1 #x25 #xf1 #x17 #x18 #x19 #x1a #x26
     #x27 #x28 #x29 #x2a #x35 #x36 #x37 #x38
     #x39 #x3a #x43 #x44 #x45 #x46 #x47 #x48
     #x49 #x4a #x53 #x54 #x55 #x56 #x57 #x58
     #x59 #x5a #x63 #x64 #x65 #x66 #x67 #x68
     #x69 #x6a #x73 #x74 #x75 #x76 #x77 #x78
     #x79 #x7a #x82 #x83 #x84 #x85 #x86 #x87
     #x88 #x89 #x8a #x92 #x93 #x94 #x95 #x96
     #x97 #x98 #x99 #x9a #xa2 #xa3 #xa4 #xa5
     #xa6 #xa7 #xa8 #xa9 #xaa #xb2 #xb3 #xb4
     #xb5 #xb6 #xb7 #xb8 #xb9 #xba #xc2 #xc3
     #xc4 #xc5 #xc6 #xc7 #xc8 #xc9 #xca #xd2
     #xd3 #xd4 #xd5 #xd6 #xd7 #xd8 #xd9 #xda
     #xe2 #xe3 #xe4 #xe5 #xe6 #xe7 #xe8 #xe9
     #xea #xf2 #xf3 #xf4 #xf5 #xf6 #xf7 #xf8
     #xf9 #xfa))

;;;Zigzag encoding matrix
(define-constant +zigzag-index+
  (2d-uint8-array '(0  1  5  6 14 15 27 28)
                  '(2  4  7 13 16 26 29 42)
                  '(3  8 12 17 25 30 41 43)
                  '(9 11 18 24 31 40 44 53)
                  '(10 19 23 32 39 45 52 54)
                  '(20 22 33 38 46 51 55 60)
                  '(21 34 37 47 50 56 59 61)
                  '(35 36 48 49 57 58 62 63)))

;;;JPEG file markers
(defconstant +M_COM+ #xfe)
(defconstant +M_SOF0+ #xc0)
(defconstant +M_SOF2+ #xc2)
(defconstant +M_DHT+ #xc4)
(defconstant +M_RST0+ #xd0)
(defconstant +M_RST7+ #xd7)
(defconstant +M_SOI+ #xd8)
(defconstant +M_EOI+ #xd9)
(defconstant +M_SOS+ #xda)
(defconstant +M_DQT+ #xdb)
(defconstant +M_DNL+ #xdc)
(defconstant +M_DRI+ #xdd)
(defconstant +M_DAC+ #xcc)
(defconstant +M_APP0+ #xe0)
(defconstant +M_APP14+ #xee)

;;; Default quantization tables
(define-constant +q-luminance+
  (2d-uint8-array '(16 11 10 16 24 40 51 61)
                  '(12 12 14 19 26 58 60 55)
                  '(14 13 16 24 40 57 69 56)
                  '(14 17 22 29 51 87 80 62)
                  '(18 22 37 56 68 109 103 77)
                  '(24 35 55 64 81 104 113 92)
                  '(49 64 78 87 103 121 120 101)
                  '(72 92 95 98 112 100 103 99)))

(define-constant +q-chrominance+
  (2d-uint8-array '(17 18 24 47 99 99 99 99)
                  '(18 21 26 66 99 99 99 99)
                  '(24 26 56 99 99 99 99 99)
                  '(47 66 99 99 99 99 99 99)
                  '(99 99 99 99 99 99 99 99)
                  '(99 99 99 99 99 99 99 99)
                  '(99 99 99 99 99 99 99 99)
                  '(99 99 99 99 99 99 99 99)))

(define-constant +q-luminance-hi+
  (2d-uint8-array '(10 7 6 10 15 25 32 38)
                  '(8 8 9 12 16 36 38 34)
                  '(9 8 10 15 25 36 43 35)
                  '(9 11 14 18 32 54 50 39)
                  '(11 14 23 35 42 68 64 48)
                  '(15 22 34 40 51 65 71 58)
                  '(31 40 49 54 64 76 75 63)
                  '(45 58 59 61 70 62 64 62)))

(define-constant +q-chrominance-hi+
  (2d-uint8-array '(11 11 15 29 62 62 62 62)
                  '(11 13 16 41 62 62 62 62)
                  '(15 16 35 62 62 62 62 62)
                  '(29 41 62 62 62 62 62 62)
                  '(62 62 62 62 62 62 62 62)
                  '(62 62 62 62 62 62 62 62)
                  '(62 62 62 62 62 62 62 62)
                  '(62 62 62 62 62 62 62 62)))

(defconstant +max-sample+ 255)

)

;;; Fast exact integer rounding (avoids SB-KERNEL::ROUND1 overhead).
;;; Precondition: |val| < most-positive-fixnum / 2 to avoid overflow in (ash r 1).
;;; In CL-JPEG quantization, val is DCT output (|val| <= 2048) and qc <= 255, so r < 255.
(declaim (inline fast-round))
(defun fast-round (val qc)
  (declare #.*optimize*
           (type (signed-byte 16) val)
           (type (unsigned-byte 8) qc))
  (multiple-value-bind (q r) (truncate val qc)
    (declare (type (signed-byte 16) q r))
    (let ((abs2r (the (unsigned-byte 16) (abs (the fixnum (ash r 1))))))
      (declare (type (unsigned-byte 16) abs2r))
      (cond ((< abs2r qc) q)
            ((> abs2r qc) (the (signed-byte 16) (if (minusp r) (1- q) (1+ q))))
            (t (if (oddp q)
                   (the (signed-byte 16) (if (minusp r) (1- q) (1+ q)))
                   q))))))

(define-constant +q-tables+ (vector +q-luminance+ +q-chrominance+))

;;; This table is used to map coefficients into SSSS value
(define-constant +csize+ (make-array 2047
                                 :initial-contents
                                 (loop for i fixnum from 0 to 2046
                                       collecting (integer-length (abs (minus i 1023))))))

;;; Some constants for colorspace mapper
(defconstant shift (1- (integer-length (ash most-positive-fixnum -7))))
(defconstant +.299+ (round (+ (* 0.299 (ash 1 shift)) 0.5)))
(defconstant +.587+ (round (+ (* 0.587 (ash 1 shift)) 0.5)))
(defconstant +.114+ (round (+ (* 0.114 (ash 1 shift)) 0.5)))
(defconstant +-.1687+ (round (+ (* -0.1687 (ash 1 shift)) 0.5)))
(defconstant +-.3313+ (round (+ (* -0.3313 (ash 1 shift)) 0.5)))
(defconstant +-.4187+ (round (+ (* -0.4187 (ash 1 shift)) 0.5)))
(defconstant +-.0813+ (round (+ (* -0.0813 (ash 1 shift)) 0.5)))
(defconstant +.5+ (round (+ (* 0.5 (ash 1 shift)) 0.5)))
(defconstant +uvoffset+ (ash 128 shift))
(defconstant +one-half+ (1- (ash 1 (1- shift))))
(defconstant +r-y-off+ 0)
(defconstant +g-y-off+ 256)
(defconstant +b-y-off+ (* 2 256))
(defconstant +r-u-off+ (* 3 256))
(defconstant +g-u-off+ (* 4 256))
(defconstant +b-u-off+ (* 5 256))
(defconstant +r-v-off+ +b-u-off+)
(defconstant +g-v-off+ (* 6 256))
(defconstant +b-v-off+ (* 7 256))

(declaim (type fixnum-array +ctab+ +cr-r-tab+ +cb-g-tab+ +cr-g-tab+ +cb-b-tab+))

;;;Direct color conversion table
(define-constant +ctab+
    (let ((table (make-array 2048 :element-type 'fixnum :initial-element 0)))
      (loop for i fixnum from 0 to 255 do
           (setf (aref table (plus i +r-y-off+))
                 (mul +.299+ i))
           (setf (aref table (plus i +g-y-off+))
                 (mul +.587+ i))
           (setf (aref table (plus i +b-y-off+))
                 (mul +.114+ i))
           (setf (aref table (plus i +r-u-off+))
                 (mul +-.1687+ i))
           (setf (aref table (plus i +g-u-off+))
                 (mul +-.3313+ i))
           (setf (aref table (plus i +b-u-off+))
                 (+ (mul +.5+ i) +uvoffset+ +one-half+))
           (setf (aref table (plus i +r-v-off+))
                 (+ (mul +.5+ i) +uvoffset+ +one-half+))
           (setf (aref table (plus i +g-v-off+))
                 (mul +-.4187+ i))
           (setf (aref table (plus i +b-v-off+))
                 (mul +-.0813+ i)))
      table))

;;; Constantsants for the inverse colorspace conversion
(defconstant +1.40200+ (round (+ (* 1.40200 (ash 1 shift)) 0.5)))
(defconstant +1.77200+ (round (+ (* 1.77200 (ash 1 shift)) 0.5)))
(defconstant +-0.71414+ (round (+ (* -0.71414 (ash 1 shift)) 0.5)))
(defconstant +-0.34414+ (round (+ (* -0.34414 (ash 1 shift)) 0.5)))

;;; Inverse color conversion tables
(define-constant +cr-r-tab+ (make-array 256
                                        :element-type 'fixnum
                                        :initial-contents
                                        (loop for i from 0 to 255
                                              for x from -128
                                              collect (ash (plus (mul +1.40200+ x) +one-half+) (- shift)))))
(define-constant +cb-g-tab+ (make-array 256
                                        :element-type 'fixnum
                                        :initial-contents
                                        (loop for i from 0 to 255
                                              for x from -128
                                              collect (plus (mul +-0.34414+ x) +one-half+))))
(define-constant +cr-g-tab+ (make-array 256
                                        :element-type 'fixnum
                                        :initial-contents
                                        (loop for i from 0 to 255
                                              for x from -128
                                              collect (mul +-0.71414+ x))))
(define-constant +cb-b-tab+ (make-array 256
                                        :element-type 'fixnum
                                        :initial-contents
                                        (loop for i from 0 to 255
                                              for x from -128
                                              collect (ash (plus (mul +1.77200+ x) +one-half+) (- shift)))))

;;; Constants for LLM DCT
(defconstant dct-shift  ; defining DCT scaling
  (if (<= (integer-length most-positive-fixnum) 31)
      (minus 13 (round (minus 31 (integer-length most-positive-fixnum)) 2))
    13))

(defconstant +shift-1+ (1- dct-shift))
(defconstant +shift+1+ (1+ dct-shift))
;; Constants deviate from libjpeg-turbo in the final bit due to a legacy rounding
;; convention (round (+ (* c 8192) 0.5)), which rounds half-integers upward rather than
;; to nearest-even. For example, +FIX-0-298631336+ is 2447 where libjpeg-turbo has 2446.
;; Preserved for bit-exact compatibility with existing CL-JPEG output.
(defconstant +shift+4+ (+ dct-shift 4))
(defconstant +FIX-0-298631336+ (round (+ (* 0.298631336 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-0-390180644+ (round (+ (* 0.390180644 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-0-541196100+ (round (+ (* 0.541196100 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-0-765366865+ (round (+ (* 0.765366865 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-0-899976223+ (round (+ (* 0.899976223 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-1-175875602+ (round (+ (* 1.175875602 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-1-501321110+ (round (+ (* 1.501321110 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-1-847759065+ (round (+ (* 1.847759065 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-1-961570560+ (round (+ (* 1.961570560 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-2-053119869+ (round (+ (* 2.053119869 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-2-562915447+ (round (+ (* 2.562915447 (ash 1 dct-shift)) 0.5)))
(defconstant +FIX-3-072711026+ (round (+ (* 3.072711026 (ash 1 dct-shift)) 0.5)))

(declaim (type uint8-array *idct-limit-array*))
;;; Post-IDCT limiting array
(defvar *idct-limit-array* (make-array 512 :initial-element 0 :element-type 'uint8))
(loop for n from 0
      for i from 128 to 383 do
      (setf (aref *idct-limit-array* i) n))
(loop for i from 384 to 511 do
      (setf (aref *idct-limit-array* i) 255))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Error conditions

(define-condition jpeg-error (error)
  ())

(define-condition jpeg-encoder-error (jpeg-error)
  ())

(define-condition internal-jpeg-encoder-error (jpeg-encoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Internal error"))))

(define-condition illegal-number-of-components (jpeg-encoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Illegal number of components specified"))))

(define-condition invalid-sampling-list (jpeg-encoder-error)
  ((components :reader components :initarg :ncomp))
  (:report (lambda (condition stream)
	     (format stream "Wrong sampling list for ~D component(s)" (components condition)))))

(define-condition invalid-quantization-tables (jpeg-encoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Too many quantization tables specified"))))

(define-condition invalid-q-factor (jpeg-encoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Invalid Q factor!"))))

(define-condition invalid-sampling (jpeg-encoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Invalid sampling specification!"))))

(define-condition jpeg-decoder-error (jpeg-error)
  ())

(define-condition unsupported-jpeg-frame-marker (jpeg-decoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Unsupported marker in the frame header"))))

(define-condition unsupported-jpeg-format (jpeg-decoder-error) 
  ((code :reader marker-code :initarg :code))
  (:report (lambda (condition stream)
	     (format stream "Unsupported JPEG format: ~X" (marker-code condition)))))

(define-condition unrecognized-file-format (jpeg-decoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Unrecognized JPEG format"))))

(define-condition invalid-buffer-supplied (jpeg-decoder-error)
  ((buffer :reader buffer :initarg :buffer))
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Invalid buffer supplied: ~A" (buffer condition)))))

(define-condition unsupported-arithmetic-encoding (jpeg-decoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "Arithmetic encoding not supported"))))

(define-condition unsupported-dnl-marker (jpeg-decoder-error)
  ()
  (:report (lambda (condition stream)
	     (declare (ignorable condition))
	     (format stream "DNL marker is not supported"))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Encoder part

;;; Subsamples inbuf into outbuf using Strategy A (block-of-blocks layout)
(defun subsample (inbuf outbuf H V xlim ylim iH iV width &optional (method *chroma-subsampling*))
  (declare #.*optimize*
           (type fixnum H V xlim ylim iV iH width)
           (type (simple-array sint16 (*)) inbuf outbuf))
  (let* ((nblocks (the fixnum (* H V)))
         (stride (the fixnum (ash nblocks 3))))
    (declare (type fixnum nblocks stride))
    (if (eq method :point)
        ;; Legacy nearest-neighbor (point) subsampling
        (loop for by fixnum from 0 below V do
          (loop for bx fixnum from 0 below H
                for b fixnum = (the fixnum (+ bx (the fixnum (* by H))))
                for b_base fixnum = (the fixnum (ash b 3)) do
            (loop for y fixnum from (the fixnum (ash by 3)) by iV
                  for yp fixnum from 0 to 7
                  for row_offset fixnum = (the fixnum (+ b_base (the fixnum (* yp stride))))
                  for clamped_y fixnum = (min y ylim)
                  for in_row fixnum = (the fixnum (* clamped_y width)) do
              (loop for x fixnum from (the fixnum (ash bx 3)) by iH
                    for xp fixnum from 0 to 7
                    for clamped_x fixnum = (min x xlim) do
                (setf (aref outbuf (the fixnum (+ row_offset xp)))
                      (aref inbuf (the fixnum (+ clamped_x in_row))))))))
        ;; Box filter averaging (quality parity with cjpeg / IJG)
        (cond
          ;; 2x2 box downsampling (most common: 4:2:0 chroma)
          ((and (= iH 2) (= iV 2))
           (loop for by fixnum from 0 below V do
             (loop for bx fixnum from 0 below H
                   for b fixnum = (the fixnum (+ bx (the fixnum (* by H))))
                   for b_base fixnum = (the fixnum (ash b 3)) do
               (loop for y fixnum from (the fixnum (ash by 3)) by 2
                     for yp fixnum from 0 to 7
                     for row_offset fixnum = (the fixnum (+ b_base (the fixnum (* yp stride))))
                     for y0 fixnum = (min y ylim)
                     for y1 fixnum = (min (the fixnum (1+ y)) ylim)
                     for r0 fixnum = (the fixnum (* y0 width))
                     for r1 fixnum = (the fixnum (* y1 width))
                     with bias fixnum = 1 do
                 (loop for x fixnum from (the fixnum (ash bx 3)) by 2
                       for xp fixnum from 0 to 7
                       for x0 fixnum = (min x xlim)
                       for x1 fixnum = (min (the fixnum (1+ x)) xlim)
                       for p00 fixnum = (aref inbuf (the fixnum (+ r0 x0)))
                       for p01 fixnum = (aref inbuf (the fixnum (+ r0 x1)))
                       for p10 fixnum = (aref inbuf (the fixnum (+ r1 x0)))
                       for p11 fixnum = (aref inbuf (the fixnum (+ r1 x1)))
                       for avg fixnum = (ash (the fixnum (+ p00 p01 p10 p11 bias)) -2) do
                   (setf (aref outbuf (the fixnum (+ row_offset xp))) avg)
                   (setf bias (the fixnum (logxor bias 3))))))))

          ;; 2x1 box downsampling (4:2:2 chroma)
          ((and (= iH 2) (= iV 1))
           (loop for by fixnum from 0 below V do
             (loop for bx fixnum from 0 below H
                   for b fixnum = (the fixnum (+ bx (the fixnum (* by H))))
                   for b_base fixnum = (the fixnum (ash b 3)) do
               (loop for y fixnum from (the fixnum (ash by 3)) by 1
                     for yp fixnum from 0 to 7
                     for row_offset fixnum = (the fixnum (+ b_base (the fixnum (* yp stride))))
                     for y0 fixnum = (min y ylim)
                     for r0 fixnum = (the fixnum (* y0 width))
                     with bias fixnum = 0 do
                 (loop for x fixnum from (the fixnum (ash bx 3)) by 2
                       for xp fixnum from 0 to 7
                       for x0 fixnum = (min x xlim)
                       for x1 fixnum = (min (the fixnum (1+ x)) xlim)
                       for p0 fixnum = (aref inbuf (the fixnum (+ r0 x0)))
                       for p1 fixnum = (aref inbuf (the fixnum (+ r0 x1)))
                       for avg fixnum = (ash (the fixnum (+ p0 p1 bias)) -1) do
                   (setf (aref outbuf (the fixnum (+ row_offset xp))) avg)
                   (setf bias (the fixnum (logxor bias 1))))))))

          ;; 1x1 point (no downsampling, e.g. Y in 4:2:0 or 4:4:4)
          ((and (= iH 1) (= iV 1))
           (loop for by fixnum from 0 below V do
             (loop for bx fixnum from 0 below H
                   for b fixnum = (the fixnum (+ bx (the fixnum (* by H))))
                   for b_base fixnum = (the fixnum (ash b 3)) do
               (loop for y fixnum from (the fixnum (ash by 3)) by 1
                     for yp fixnum from 0 to 7
                     for row_offset fixnum = (the fixnum (+ b_base (the fixnum (* yp stride))))
                     for clamped_y fixnum = (min y ylim)
                     for in_row fixnum = (the fixnum (* clamped_y width)) do
                 (loop for x fixnum from (the fixnum (ash bx 3)) by 1
                       for xp fixnum from 0 to 7
                       for clamped_x fixnum = (min x xlim) do
                   (setf (aref outbuf (the fixnum (+ row_offset xp)))
                         (aref inbuf (the fixnum (+ clamped_x in_row)))))))))

          ;; General fallback for arbitrary sampling factors
          (t
           (loop for by fixnum from 0 below V do
             (loop for bx fixnum from 0 below H
                   for b fixnum = (the fixnum (+ bx (the fixnum (* by H))))
                   for b_base fixnum = (the fixnum (ash b 3)) do
               (loop for y fixnum from (the fixnum (ash by 3)) by iV
                     for yp fixnum from 0 to 7
                     for row_offset fixnum = (the fixnum (+ b_base (the fixnum (* yp stride))))
                     for clamped_y fixnum = (min y ylim)
                     for in_row fixnum = (the fixnum (* clamped_y width)) do
                 (loop for x fixnum from (the fixnum (ash bx 3)) by iH
                       for xp fixnum from 0 to 7
                       for clamped_x fixnum = (min x xlim) do
                   (setf (aref outbuf (the fixnum (+ row_offset xp)))
                         (aref inbuf (the fixnum (+ clamped_x in_row)))))))))))))


;;; Returns sum of Vi*Hi
(defun count-relation (smp)
  (loop for entry in smp
        summing (mul (first entry) (second entry))))

;;; Cutting specified part of image (used for non-RGB images)
(defun crop-image (inbuf outbuf dx dy h w height width ncomp)
  (declare #.*optimize*
           (type (simple-array (unsigned-byte 8) (*)) inbuf)
           (type (simple-array (simple-array sint16 (*)) (*)) outbuf)
           (type fixnum dx dy h w height width ncomp))
  (let ((xend (min (the fixnum (1- w)) (the fixnum (+ dx (the fixnum (1- width))))))
        (yend (min (the fixnum (1- h)) (the fixnum (+ dy (the fixnum (1- height)))))))
    (declare (type fixnum xend yend))
    (loop for yd fixnum from dy to yend
          for ypos fixnum = (the fixnum (* (the fixnum (* w yd)) ncomp))
          for cy fixnum = (the fixnum (- yd dy))
          for out_row fixnum = (the fixnum (* cy width)) do
      (loop for xd fixnum from dx to xend
            for pos fixnum = (the fixnum (+ (the fixnum (* xd ncomp)) ypos))
            for cx fixnum = (the fixnum (- xd dx))
            for out_idx fixnum = (the fixnum (+ cx out_row)) do
        (loop for i fixnum from 0 below ncomp
              for comp-buf of-type (simple-array sint16 (*)) = (aref outbuf i) do
          (setf (aref comp-buf out_idx)
                (the fixnum (- (aref inbuf (the fixnum (+ pos i))) 128))))))
    (values xend yend)))

;;; Direct color mapping into flat 1D component buffers
(defun colorspace-convert (RGB YUV dx dy h w height width)
  (declare #.*optimize*
           (type (simple-array (unsigned-byte 8) (*)) RGB)
           (type (simple-array (simple-array sint16 (*)) (*)) YUV)
           (type fixnum dx dy h w height width)
           (type fixnum-array +ctab+))
  (let ((xend (min (the fixnum (1- w)) (the fixnum (+ dx (the fixnum (1- width))))))
        (yend (min (the fixnum (1- h)) (the fixnum (+ dy (the fixnum (1- height))))))
        (Y (aref YUV 0))
        (U (aref YUV 1))
        (V (aref YUV 2)))
    (declare (type fixnum xend yend)
             (type (simple-array sint16 (*)) Y U V))
    (loop for yd fixnum from dy to yend
          for ypos fixnum = (the fixnum (* (the fixnum (* w yd)) 3))
          for cy fixnum = (the fixnum (- yd dy))
          for out_row fixnum = (the fixnum (* cy width)) do
      (loop for xd fixnum from dx to xend
            for pos fixnum = (the fixnum (+ (the fixnum (* xd 3)) ypos))
            for r fixnum = (aref RGB (the fixnum (+ pos 2)))
            for g fixnum = (aref RGB (the fixnum (1+ pos)))
            for b fixnum = (aref RGB pos)
            for cx fixnum = (the fixnum (- xd dx))
            for out_idx fixnum = (the fixnum (+ cx out_row)) do
        (setf (aref Y out_idx)
              (the fixnum (- (the fixnum (ash (the fixnum (+ (aref +ctab+ (the fixnum (+ r +r-y-off+)))
                                                             (aref +ctab+ (the fixnum (+ g +g-y-off+)))
                                                             (aref +ctab+ (the fixnum (+ b +b-y-off+)))))
                                              (- shift)))
                             128)))
        (setf (aref U out_idx)
              (the fixnum (- (the fixnum (ash (the fixnum (+ (aref +ctab+ (the fixnum (+ r +r-u-off+)))
                                                             (aref +ctab+ (the fixnum (+ g +g-u-off+)))
                                                             (aref +ctab+ (the fixnum (+ b +b-u-off+)))))
                                              (- shift)))
                             128)))
        (setf (aref V out_idx)
              (the fixnum (- (the fixnum (ash (the fixnum (+ (aref +ctab+ (the fixnum (+ r +r-v-off+)))
                                                             (aref +ctab+ (the fixnum (+ g +g-v-off+)))
                                                             (aref +ctab+ (the fixnum (+ b +b-v-off+)))))
                                              (- shift)))
                             128)))))
    (values xend yend)))

;;; Converts given image sampling into frequencies of pixels of components
(defun convert-sampling (s Hmax Vmax)
  (declare (type fixnum Hmax Vmax))
  (make-array (length s)
              :initial-contents (loop for entry in s
                                      collecting (list (the fixnum (/ Hmax (first entry)))
                                                       (the fixnum (/ Vmax (second entry)))))))

;;; Quantization (also removes factor of 8 after DCT)

;;; Quantization in Strategy A layout (also removes factor of 8 after DCT)
(defun quantize (block b nblocks q-table)
  (declare #.*optimize*
	   (type (simple-array sint16 (*)) block)
	   (type uint8-2d-array q-table)
           (type fixnum b nblocks))
  (let* ((stride (the fixnum (ash nblocks 3)))
         (b_base (the fixnum (ash b 3))))
    (declare (type fixnum stride b_base))
    (loop for r fixnum from 0 to 7
          for q-row of-type (simple-array (unsigned-byte 8) (8)) across q-table
          for row_offset fixnum = (+ b_base (the fixnum (* r stride))) do
      (loop for x fixnum from 0 to 7
            for idx fixnum = (+ row_offset x)
            for val fixnum = (ash (aref block idx) -3)
            for qc fixnum = (aref q-row x) do
        (setf (aref block idx) (the fixnum (fast-round val qc)))))))

;;; LLM DCT aux definitions
(defun descale (x n)
  (declare #.*optimize* (type fixnum x n))
  (the fixnum (ash (plus x (ash 1 (1- n))) (- n))))

;;; Implementation of Loeffer, Ligtenberg and Moschytz forward DCT in Strategy A layout
(defun llm-dct (data b nblocks)
  (declare #.*optimize*
           (type (simple-array sint16 (*)) data)
           (type fixnum b nblocks))
  (let* ((stride (the fixnum (ash nblocks 3)))
         (b_base (the fixnum (ash b 3))))
    (declare (type fixnum stride b_base))
    (loop with tmp0 fixnum and tmp1 fixnum and tmp2 fixnum
          and tmp3 fixnum and tmp4 fixnum and tmp5 fixnum
          and tmp6 fixnum and tmp7 fixnum and tmp10 fixnum
          and tmp11 fixnum and tmp12 fixnum and tmp13 fixnum
          and z1 fixnum and z2 fixnum and z3 fixnum
          and z4 fixnum and z5 fixnum do
      ;; Pass 1: Rows
      (loop for r fixnum from 0 to 7
            for row_offset fixnum = (the fixnum (+ b_base (the fixnum (* r stride)))) do
        (let ((d0 (the fixnum (+ row_offset 0)))
              (d1 (the fixnum (+ row_offset 1)))
              (d2 (the fixnum (+ row_offset 2)))
              (d3 (the fixnum (+ row_offset 3)))
              (d4 (the fixnum (+ row_offset 4)))
              (d5 (the fixnum (+ row_offset 5)))
              (d6 (the fixnum (+ row_offset 6)))
              (d7 (the fixnum (+ row_offset 7))))
          (declare (type fixnum d0 d1 d2 d3 d4 d5 d6 d7))
          (setf tmp0 (+ (aref data d0) (aref data d7)))
          (setf tmp7 (- (aref data d0) (aref data d7)))
          (setf tmp1 (+ (aref data d1) (aref data d6)))
          (setf tmp6 (- (aref data d1) (aref data d6)))
          (setf tmp2 (+ (aref data d2) (aref data d5)))
          (setf tmp5 (- (aref data d2) (aref data d5)))
          (setf tmp3 (+ (aref data d3) (aref data d4)))
          (setf tmp4 (- (aref data d3) (aref data d4)))
          (setf tmp10 (+ tmp0 tmp3))
          (setf tmp13 (- tmp0 tmp3))
          (setf tmp11 (+ tmp1 tmp2))
          (setf tmp12 (- tmp1 tmp2))
          (setf (aref data d0) (ash (+ tmp10 tmp11) 1))
          (setf (aref data d4) (ash (- tmp10 tmp11) 1))
          (setf z1 (mul (+ tmp12 tmp13) +fix-0-541196100+))
          (setf (aref data d2) (descale (+ z1 (mul tmp13 +fix-0-765366865+)) +shift-1+))
          (setf (aref data d6) (descale (+ z1 (mul tmp12 (- +fix-1-847759065+))) +shift-1+))
          (setf z1 (+ tmp4 tmp7))
          (setf z2 (+ tmp5 tmp6))
          (setf z3 (+ tmp4 tmp6))
          (setf z4 (+ tmp5 tmp7))
          (setf z5 (mul (+ z3 z4) +fix-1-175875602+))
          (setf tmp4 (mul tmp4 +fix-0-298631336+))
          (setf tmp5 (mul tmp5 +fix-2-053119869+))
          (setf tmp6 (mul tmp6 +fix-3-072711026+))
          (setf tmp7 (mul tmp7 +fix-1-501321110+))
          (setf z1 (mul z1 (- +fix-0-899976223+)))
          (setf z2 (mul z2 (- +fix-2-562915447+)))
          (setf z3 (mul z3 (- +fix-1-961570560+)))
          (setf z4 (mul z4 (- +fix-0-390180644+)))
          (incf z3 z5)
          (incf z4 z5)
          (setf (aref data d7) (descale (plus3 tmp4 z1 z3) +shift-1+))
          (setf (aref data d5) (descale (plus3 tmp5 z2 z4) +shift-1+))
          (setf (aref data d3) (descale (plus3 tmp6 z2 z3) +shift-1+))
          (setf (aref data d1) (descale (plus3 tmp7 z1 z4) +shift-1+))))

      ;; Pass 2: Columns
      (loop for c fixnum from 7 downto 0
            for col_base fixnum = (the fixnum (+ b_base c)) do
        (let ((d0 (the fixnum (+ col_base (the fixnum (* 0 stride)))))
              (d1 (the fixnum (+ col_base (the fixnum (* 1 stride)))))
              (d2 (the fixnum (+ col_base (the fixnum (* 2 stride)))))
              (d3 (the fixnum (+ col_base (the fixnum (* 3 stride)))))
              (d4 (the fixnum (+ col_base (the fixnum (* 4 stride)))))
              (d5 (the fixnum (+ col_base (the fixnum (* 5 stride)))))
              (d6 (the fixnum (+ col_base (the fixnum (* 6 stride)))))
              (d7 (the fixnum (+ col_base (the fixnum (* 7 stride))))))
          (declare (type fixnum d0 d1 d2 d3 d4 d5 d6 d7))
          (setf tmp0 (+ (aref data d0) (aref data d7)))
          (setf tmp7 (- (aref data d0) (aref data d7)))
          (setf tmp1 (+ (aref data d1) (aref data d6)))
          (setf tmp6 (- (aref data d1) (aref data d6)))
          (setf tmp2 (+ (aref data d2) (aref data d5)))
          (setf tmp5 (- (aref data d2) (aref data d5)))
          (setf tmp3 (+ (aref data d3) (aref data d4)))
          (setf tmp4 (- (aref data d3) (aref data d4)))
          (setf tmp10 (+ tmp0 tmp3))
          (setf tmp13 (- tmp0 tmp3))
          (setf tmp11 (+ tmp1 tmp2))
          (setf tmp12 (- tmp1 tmp2))
          (setf (aref data d0) (descale (+ tmp10 tmp11) 1))
          (setf (aref data d4) (descale (- tmp10 tmp11) 1))
          (setf z1 (mul (+ tmp12 tmp13) +fix-0-541196100+))
          (setf (aref data d2) (descale (+ z1 (mul tmp13 +fix-0-765366865+)) +shift+1+))
          (setf (aref data d6) (descale (+ z1 (mul tmp12 (- +fix-1-847759065+))) +shift+1+))
          (setf z1 (+ tmp4 tmp7))
          (setf z2 (+ tmp5 tmp6))
          (setf z3 (+ tmp4 tmp6))
          (setf z4 (+ tmp5 tmp7))
          (setf z5 (mul (+ z3 z4) +fix-1-175875602+))
          (setf tmp4 (mul tmp4 +fix-0-298631336+))
          (setf tmp5 (mul tmp5 +fix-2-053119869+))
          (setf tmp6 (mul tmp6 +fix-3-072711026+))
          (setf tmp7 (mul tmp7 +fix-1-501321110+))
          (setf z1 (mul z1 (- +fix-0-899976223+)))
          (setf z2 (mul z2 (- +fix-2-562915447+)))
          (setf z3 (mul z3 (- +fix-1-961570560+)))
          (setf z4 (mul z4 (- +fix-0-390180644+)))
          (incf z3 z5)
          (incf z4 z5)
          (setf (aref data d7) (descale (plus3 tmp4 z1 z3) +shift+1+))
          (setf (aref data d5) (descale (plus3 tmp5 z2 z4) +shift+1+))
          (setf (aref data d3) (descale (plus3 tmp6 z2 z3) +shift+1+))
          (setf (aref data d1) (descale (plus3 tmp7 z1 z4) +shift+1+))))
      (return))))

;;; Forward DCT and quantization in Strategy A layout
(defun crunch (buf pos nblocks table)
  (declare #.*optimize*
           (type fixnum pos nblocks)
           (type (simple-array sint16 (*)) buf))
  (if (and (not (eql *use-simd* :off)) *llm-dct-impl*)
      (funcall (the function *llm-dct-impl*) buf pos nblocks)
      (llm-dct buf pos nblocks))
  (quantize buf pos nblocks table))

(defconstant +max-qval+ 255
  "Maximum quantization table entry value for 8-bit sample precision (ITU-T T.81 Table B.1).")

;;; Q-tables scaling (deterministic integer math)
(defun q-scale (table q-factor)
  (declare #.*optimize*
	   (type uint8-2d-array table)
	   (type uint8 q-factor))
  (when (/= q-factor 64)
    (loop for q-row of-type uint8-array across table do
          (loop for x fixnum from 0 to 7 do
                (setf (aref q-row x)
                      (max 1 (min +max-qval+ (the fixnum (round (* (aref q-row x) q-factor) 64)))))))))

;;; Function that maps value into SSSS
(defun csize (n)
    (declare #.*optimize* (type fixnum n))
    (aref +csize+ (plus n 1023)))

;;; zigzag ordering in Strategy A layout
(defun zigzag (buffer b nblocks zz-result)
  (declare #.*optimize*
	   (type (simple-array sint16 (*)) buffer)
           (type (simple-array sint16 (64)) zz-result)
           (type fixnum b nblocks))
  (let* ((stride (the fixnum (ash nblocks 3)))
         (b_base (the fixnum (ash b 3))))
    (declare (type fixnum stride b_base))
    (loop for r fixnum from 0 to 7
          for z-row of-type (simple-array (unsigned-byte 8) (8)) across +zigzag-index+
          for row_offset fixnum = (+ b_base (the fixnum (* r stride))) do
      (loop for x fixnum from 0 to 7 do
        (setf (aref zz-result (aref z-row x))
              (aref buffer (+ row_offset x)))))
    zz-result))

(defun zigzag8 (buffer zz-result)
  (declare #.*optimize*
           (type uint8-2d-array buffer)
           (type uint8-array zz-result))
  (loop for row of-type uint8-array across buffer
     for z-row of-type uint8-array across +zigzag-index+ do
       (loop for x fixnum from 0 to 7 do
            (setf (aref zz-result (aref z-row x))
                  (the fixnum (aref row x)))))
  zz-result)

;;; Writes frame header
(defun write-frame-header (maxX maxY cn q-tables sampling tqv out-stream)
  (declare #.*optimize* (type fixnum maxX maxY cn)
	   (type (simple-array uint8-2d-array (*)) q-tables)
	   ;;(type uint8-array tqv)
	   )
  (write-huffman-tables out-stream)
  (write-quantization-tables q-tables out-stream)
  ;; writing frame header
  (write-marker +M_SOF0+ out-stream)
  (write-byte 0 out-stream) ; length
  (write-byte (plus 8 (mul 3 cn)) out-stream)
  (write-byte 8 out-stream) ; sample precision
  (write-byte (ash maxY -8) out-stream) ; max height
  (write-byte (logand maxY #xff) out-stream)
  (write-byte (ash maxX -8) out-stream) ; max width
  (write-byte (logand maxX #xff) out-stream)
  (write-byte cn out-stream) ; number of components
  (loop for entry in sampling
        for i fixnum from 0 by 1 do
        (write-byte i out-stream)
        (write-byte         ; H and V
         (deposit-field (second entry) (byte 4 0)(ash (first entry) 4))
         out-stream)
        (write-byte (aref tqv i) out-stream))) ; Tq

;;; Writes byte with stuffing (adds zero after FF code)
(defun write-stuffed (b s)
  (declare #.*optimize* (type uint8 b)
           (type stream s))
   (write-byte b s)
   (if (= b #xFF)
      (write-byte 0 s)))

;;; A function for bit streaming
;;; NB: probably it's a good idea to encapsulate this behavior into a class, but I'm
;;; afraid that method dispatch would be too slow

(declaim (type (simple-array fixnum (17)) +bit-masks+))
(defparameter +bit-masks+
  (let ((arr (make-array 17 :element-type 'fixnum)))
    (dotimes (i 17)
      (setf (aref arr i) (1- (ash 1 i))))
    arr))

(defstruct (write-bits-state (:conc-name write-bits-))
  (buffer 0 :type fixnum)
  (bits 0 :type fixnum)
  (out-buffer (make-array 4096 :element-type 'uint8) :type (simple-array uint8 (4096)))
  (out-pos 0 :type fixnum))

(declaim (inline write-bits))
(declaim (inline write-bits))
(defun write-bits (bi ni s write-bits-state)
  (declare #.*optimize*
           (type fixnum bi)
           (type (integer 0 16) ni)
           (type stream s)
           (type write-bits-state write-bits-state))
  (when (> ni 0)
    (let ((buf (the (unsigned-byte 64)
                    (logand (logior (ash (write-bits-buffer write-bits-state) ni)
                                    (logand bi (aref +bit-masks+ ni)))
                            #xffffffffffffffff)))
          (bits (+ (write-bits-bits write-bits-state) ni))
          (out-buf (write-bits-out-buffer write-bits-state))
          (out-pos (write-bits-out-pos write-bits-state)))
      (declare (type (unsigned-byte 64) buf)
               (type fixnum bits out-pos)
               (type (simple-array uint8 (4096)) out-buf))
      (loop while (>= bits 8) do
        (decf bits 8)
        (let ((byte (logand (ash buf (- bits)) #xff)))
          (declare (type uint8 byte))
          (setf (aref out-buf out-pos) byte)
          (incf out-pos)
          (when (= byte 255)
            (when (= out-pos 4096)
              (write-sequence out-buf s)
              (setf out-pos 0))
            (setf (aref out-buf out-pos) 0)
            (incf out-pos))
          (when (= out-pos 4096)
            (write-sequence out-buf s)
            (setf out-pos 0))))
      (setf (write-bits-buffer write-bits-state) (logand buf (aref +bit-masks+ bits))
            (write-bits-bits write-bits-state) bits
            (write-bits-out-pos write-bits-state) out-pos))))

;;; Encodes block using specified huffman tables, returns new pred (DC prediction value)
(defun encode-block (block tables pred s write-bits-state)
  (declare #.*optimize* (type fixnum pred)
           (type sint16-array block)
           (type stream s)
           (type write-bits-state write-bits-state))
  (let* ((ehufpacked-dc (third (first tables)))
         (ehufpacked-ac (third (second tables)))
         (newpred (aref block 0))
         (diff (minus newpred pred))
         (dcpos (csize diff))
         (buf (write-bits-buffer write-bits-state))
         (bits (write-bits-bits write-bits-state))
         (out-buf (write-bits-out-buffer write-bits-state))
         (out-pos (write-bits-out-pos write-bits-state)))
    (declare (type fixnum pred newpred diff dcpos bits out-pos)
             (type (unsigned-byte 64) buf)
	     (type fixnum-array ehufpacked-ac ehufpacked-dc)
             (type (simple-array uint8 (4096)) out-buf))
    (macrolet ((put-bits (bi ni)
                 `(let ((n ,ni))
                    (declare (type (integer 0 16) n))
                    (when (> n 0)
                      (setf buf (the (unsigned-byte 64)
                                     (logand (logior (ash buf n)
                                                     (logand ,bi (aref +bit-masks+ n)))
                                             #xffffffffffffffff)))
                      (incf bits n)
                      (loop while (>= bits 8) do
                        (decf bits 8)
                        (let ((b (logand (ash buf (- bits)) #xff)))
                          (declare (type uint8 b))
                          (setf (aref out-buf out-pos) b)
                          (incf out-pos)
                          (when (= b 255)
                            (when (= out-pos 4096)
                              (write-sequence out-buf s)
                              (setf out-pos 0))
                            (setf (aref out-buf out-pos) 0)
                            (incf out-pos))
                          (when (= out-pos 4096)
                            (write-sequence out-buf s)
                            (setf out-pos 0))))
                      (setf buf (logand buf (aref +bit-masks+ bits))))))
               (put-packed (entry)
                 `(let ((e (the fixnum ,entry)))
                    (put-bits (ash e -5) (logand e 31)))))
      ;; writing dc code first via single packed lookup
      (put-packed (aref ehufpacked-dc dcpos))
      ;; Note: for negative difference, JPEG standard emits (1- diff) in (csize diff) bits.
      (if (minusp diff)
          (put-bits (1- diff) (csize diff))
          (put-bits diff (csize diff)))
      ;; writing ac sequence with last-nonzero coefficient pre-scan (skips trailing zero search)
      (let ((last-ac (loop for k fixnum from 63 downto 1
                           when (/= (aref block k) 0)
                           return k
                           finally (return 0))))
        (declare (type (integer 0 63) last-ac))
        (if (zerop last-ac)
            (put-packed (aref ehufpacked-ac 0)) ; writing EOB
            (progn
              (loop with r fixnum = 0 for k fixnum from 1 to last-ac do
                (let ((val (aref block k)))
                  (declare (type fixnum val))
                  (if (zerop val)
                      (incf r)
                      (progn
                        (loop while (> r 15) do
                          (put-packed (aref ehufpacked-ac #xf0)) ; ZRL
                          (decf r 16))
                        (let* ((ssss (csize val))
                               (rs (plus ssss (ash r 4))))
                          (put-packed (aref ehufpacked-ac rs))
                          ;; Do not mutate BLOCK. The previous implementation did (DECF (AREF BLOCK K) 1)
                          ;; here; instead we pass (1- val) to emit standard JPEG sign-magnitude bits
                          ;; without memory writebacks.
                          (if (minusp val)
                              (put-bits (1- val) ssss)
                              (put-bits val ssss)))
                        (setf r 0)))))
              (when (< last-ac 63)
                (put-packed (aref ehufpacked-ac 0)))))))
    (setf (write-bits-buffer write-bits-state) buf
          (write-bits-bits write-bits-state) bits
          (write-bits-out-pos write-bits-state) out-pos)
    newpred))

;;; Emits q-tables
(defun write-quantization-tables (tables s)
  (let ((len (plus 2 (mul 65 (length tables))))
        (zz-result (make-array 64 :element-type 'uint8)))
    (write-marker +M_DQT+ s)
    (write-byte (ash len -8) s) ; MSB
    (write-byte (logand len #xff) s) ; LSB
    (loop for table across tables
          for i fixnum from 0 do
          (write-byte i s)
          (write-sequence (zigzag8 table zz-result) s))))

;;; Emits huffman tables in the following order:
;;; luminance DC
;;; luminance AC
;;; chrominance DC
;;; chrominance AC
(defun write-huffman-tables (s)
  (let ((len (+ 2 (* 17 4)
                (length +luminance-dc-values+)
                (length +luminance-ac-values+)
                (length +chrominance-dc-values+)
                (length +chrominance-ac-values+))))
    (write-marker +M_DHT+ s)
    (write-byte (ash len -8) s) ; MSB
    (write-byte (logand len #xff) s) ; LSB
    (write-hufftable +luminance-dc-bits+ +luminance-dc-values+ 0 s)
    (write-hufftable +luminance-ac-bits+ +luminance-ac-values+ 16 s)
    (write-hufftable +chrominance-dc-bits+ +chrominance-dc-values+ 1 s)
    (write-hufftable +chrominance-ac-bits+ +chrominance-ac-values+ 17 s)))

;;; Writes single huffman table
(defun write-hufftable (bits vals tcti s)
  (declare (type fixnum tcti))
    (write-byte tcti s) ; Tc/Th
    (write-sequence bits s)
    (write-sequence vals s))

;;; Drops specified marker into the stream
(defun write-marker (code to)
   (write-byte #xFF to)
   (write-byte code to))

;;; Writing some markers into the stream
(defun prepare-JFIF-stream (out-stream)
   (write-marker +M_SOI+ out-stream)
   (write-marker +M_APP0+ out-stream)
   (write-byte 0 out-stream) ; length
   (write-byte 16 out-stream)
   (write-byte #x4a out-stream)
   (write-byte #x46 out-stream)
   (write-byte #x49 out-stream)
   (write-byte #x46 out-stream)
   (write-byte 0 out-stream)
   (write-byte 1 out-stream) ; version
   (write-byte 2 out-stream)
   (write-byte 0 out-stream) ; units
   (write-byte 0 out-stream) ; density
   (write-byte 1 out-stream)
   (write-byte 0 out-stream)
   (write-byte 1 out-stream)
   (write-byte 0 out-stream) ; thumbnail
   (write-byte 0 out-stream))

;;; Builds common decoding and encoding tables
(defun build-universal-tables (bits)
  (let ((huffsize (make-array 256 :element-type 'fixnum))
        (huffcode (make-array 256 :element-type 'fixnum))
        (lastk 0))
    (declare #.*optimize* (type fixnum lastk)
             (type fixnum-array huffcode huffsize)
	     (type uint8-array bits))
    ;; generating huffsize
      (loop for i fixnum from 1 to 16
            with k fixnum = 0 and j fixnum = 1 do
            (loop until (> j (aref bits (1- i))) do
                  (setf (aref huffsize k) i)
                  (incf k)
                  (incf j)
                  finally (setf j 1))
            finally (progn (setf lastk k) (setf (aref huffsize lastk) 0)))

      ;; generating huffcode
      (loop with k fixnum = 0 and code fixnum = 0 and si fixnum = (aref huffsize 0) do
            (loop do
                  (setf (aref huffcode k) code)
                  (incf code)
                  (incf k)
                  when (/= (aref huffsize k) si) do (return))
            when (zerop (aref huffsize k)) do
            (return)
            else do
            (loop do
                  (setf code (ash code 1))
                  (incf si)
                  when (= (aref huffsize k) si) do (return)))
      (values huffcode huffsize lastk)))

;;;Builds ordered code tables for encoder (with packed (code << 5 | size) entry)
(defun build-tables (bits huffval)
  (let ((ehufco (make-array 256 :element-type 'fixnum))
        (ehufsi (make-array 256 :element-type 'fixnum))
        (ehufpacked (make-array 256 :element-type 'fixnum)))
    (multiple-value-bind (huffcode huffsize lastk)
        (build-universal-tables bits)
      (declare (type fixnum-array huffsize huffcode)
               (type fixnum lastk))
      (loop with i fixnum for k from 0 below lastk do
            (setf i (aref huffval k))
            (setf (aref ehufco i) (aref huffcode k))
            (setf (aref ehufsi i) (aref huffsize k))
            (setf (aref ehufpacked i) (the fixnum (logior (ash (aref huffcode k) 5) (aref huffsize k)))))
      (list ehufsi ehufco ehufpacked))))

;;; Main encoder function (user interface)
(defun encode-image-stream (out-stream image ncomp h w
                            &key (q-tabs +q-tables+) (sampling '((2 2)(1 1)(1 1))) (q-factor 64)
                                 (chroma-subsampling *chroma-subsampling*))
  (declare #.*optimize*
           (type fixnum ncomp h w q-factor)
           (type (member :box :point) chroma-subsampling)
           (type uint8-array image))
  (when (= ncomp 1)
    (setq sampling '((1 1))))
  (let* ((wd (loop for entry in sampling maximize (the fixnum (first entry))))
         (ht (loop for entry in sampling maximize (the fixnum (second entry))))
	 (zz-result (make-array 64 :element-type 'sint16))
         (write-bits-state (make-write-bits-state))
         (isampling (convert-sampling sampling wd ht))
         (height (the fixnum (ash ht 3)))
         (width (the fixnum (ash wd 3)))
         (mcu-size (* height width))
         (YUV (make-array ncomp
			  :element-type '(simple-array sint16 (*))
                          :initial-contents
                          (loop for i fixnum from 0 below ncomp collecting
                               (make-array (+ mcu-size 32) :element-type 'sint16 :initial-element 0))))
         (sampled-buf (make-array (+ (* (* ht wd) 64) 32)
				  :element-type 'sint16
                                  :initial-element 0))
         (preds (make-array ncomp :initial-element 0))
         (tqv (case ncomp
                (3 #(0 1 1)) ; q-tables destinations for various component numbers
                (1 #(0))
                (2 #(0 1))
                (4 #(0 1 2 3))
                (otherwise (error 'illegal-number-of-components)))))
    (declare (type (simple-array (simple-array sint16 (*)) (*)) YUV)
             (type (simple-array sint16 (*)) sampled-buf))
    (cond ((/= ncomp (length sampling))
           (error 'invalid-sampling-list :ncomp ncomp))
          ((> (length q-tabs) ncomp)
           (error 'invalid-quantization-tables))
          ((zerop q-factor)
           (error 'invalid-q-factor))
          ((> (count-relation sampling) 10)
           (error 'invalid-sampling)))
    (when (/= q-factor 64)
      (let ((q-tabs2 (make-array (length q-tabs)
				 :element-type 'uint8-2d-array
                                 :initial-contents
                                 (loop for k fixnum from 0 below (length q-tabs)
                                    collecting (make-array 8 :element-type 'uint8-array
							   :initial-contents
                                                           (loop for i fixnum from 0 to 7
                                                              collecting (make-array 8 :element-type 'uint8)))))))
	(declare (type (simple-array uint8-2d-array (*)) q-tabs2))
        (loop for entry across q-tabs
           for entry2 across q-tabs2 do
           (loop for x fixnum from 0 to 7 do
                (loop for y fixnum from 0 to 7 do
                     (setf (u8ref entry2 x y) (the fixnum (u8ref entry x y))))))
        (setq q-tabs q-tabs2))
      (loop for entry across q-tabs do  ; scaling all q-tables
           (q-scale entry q-factor)))
    (if (and (/= ncomp 1) (/= ncomp 3))
        (write-marker +M_SOI+ out-stream)
        (prepare-JFIF-stream out-stream))
    (write-frame-header w h ncomp q-tabs sampling tqv out-stream) ; frame header
    ;; writing scan header
    (write-marker +M_SOS+ out-stream)
    (write-byte 0 out-stream)           ; length
    (write-byte (plus 6 (ash ncomp 1)) out-stream)
    (write-byte ncomp out-stream)    ; number of components in the scan
    (loop for Cj from 0 below ncomp do
         (write-byte Cj out-stream)     ; component ID
         (write-byte (if (zerop Cj) 0 17) out-stream)) ; TdTa
    (write-byte 0 out-stream)                          ; Ss
    (write-byte 63 out-stream)                         ; Se
    (write-byte 0 out-stream)                          ; AhAl

    (let ((luminance-tabset (list
                             (build-tables +luminance-dc-bits+ +luminance-dc-values+)
                             (build-tables +luminance-ac-bits+ +luminance-ac-values+)))
          (chrominance-tabset (list (build-tables +chrominance-dc-bits+ +chrominance-dc-values+)
                                    (build-tables +chrominance-ac-bits+ +chrominance-ac-values+))))
      (loop for dy fixnum from 0 below h by height do
           (loop for dx fixnum from 0 below w by width do
                (multiple-value-bind (xlim ylim)
                    (if (= ncomp 3)
                       (colorspace-convert image YUV dx dy h w height width)
                        (crop-image image YUV dx dy h w height width ncomp))
                  (declare (type fixnum xlim ylim)
                           (dynamic-extent xlim ylim))
                  (loop for comp across YUV
                     for freq in sampling
                     for ifreq across isampling
                     for iH fixnum = (first ifreq)
                     for iV fixnum = (second ifreq)
                     for cn fixnum from 0
                     for hufftabs = (if (zerop cn)
                                        luminance-tabset
                                        chrominance-tabset)
                     ;; choosing appropriate q-table for a component
                     for q-tab = (aref q-tabs (aref tqv cn))
                     for comp-h fixnum = (first freq)
                     for comp-v fixnum = (second freq)
                     for nblocks fixnum = (* comp-h comp-v) do
                     (subsample comp sampled-buf comp-h comp-v (minus xlim dx) (minus ylim dy) iH iV width chroma-subsampling)
                     (loop for y fixnum from 0 below comp-v
                        for ypos fixnum = (if (> (plus dy (ash y 3)) ylim)
                                              (mul (rem (ash ylim -3) comp-v) comp-h)
                                              (mul y comp-h)) do
                        (loop for x fixnum from 0 below comp-h
                           for pos fixnum = (if (> (plus dx (ash x 3)) xlim)
                                                (plus (rem (ash xlim -3) comp-h) ypos)
                                                (plus x ypos)) do
                           (crunch sampled-buf pos nblocks q-tab)
                           (setf (aref preds cn)
                                 (encode-block (zigzag sampled-buf pos nblocks zz-result)
                                               hufftabs (aref preds cn) out-stream
                                               write-bits-state)))))))))
    (let ((bits (write-bits-bits write-bits-state))
          (buf (write-bits-buffer write-bits-state))
          (out-buf (write-bits-out-buffer write-bits-state))
          (out-pos (write-bits-out-pos write-bits-state)))
      (declare (type fixnum bits buf out-pos)
               (type (simple-array uint8 (4096)) out-buf))
      (unless (zerop bits)
        (let ((padded (the fixnum (logior (the fixnum (ash buf (the fixnum (- 8 bits))))
                                          (the fixnum (ldb (byte (- 8 bits) 0) #xff))))))
          (setf (aref out-buf out-pos) padded)
          (incf out-pos)
          (when (= padded 255)
            (when (= out-pos 4096)
              (write-sequence out-buf out-stream)
              (setf out-pos 0))
            (setf (aref out-buf out-pos) 0)
            (incf out-pos))
          (when (= out-pos 4096)
            (write-sequence out-buf out-stream)
            (setf out-pos 0))))
      (when (> out-pos 0)
        (write-sequence out-buf out-stream :end out-pos)
        (setf (write-bits-out-pos write-bits-state) 0)))
    (write-marker +M_EOI+ out-stream)))

(defun encode-image (filename image ncomp h w &rest args)
  (with-open-file (out-stream filename
                              :direction :output
                              :element-type 'uint8
                              :if-exists :supersede)
    (apply #'encode-image-stream out-stream image ncomp h w args)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Decoder part

;;; Contains all information about a single scan
(defstruct scan
  (ncomp 1 :type fixnum) ; number of image components in the scan
  (x 0 :type fixnum) ; current processing
  (y 0 :type fixnum) ; positions
  (cdesc (make-array 4
                     :initial-contents (loop repeat 4 collect (list 0 0)))
         :type (simple-array t (*))) ; descriptors of all components in the scan
  (ss 0 :type fixnum)   ; spectral selection start
  (se 63 :type fixnum)  ; spectral selection end
  (ah 0 :type fixnum)   ; successive approximation high
  (al 0 :type fixnum))  ; successive approximation low

;;; Contains huffman decoding tables
(defstruct huffstruct
  mincode
  maxcode
  (bits (make-array 16 :element-type 'uint8) :type uint8-array)
  huffval
  huffcode
  valptr
  (lookup (make-array 256 :element-type '(unsigned-byte 16) :initial-element (ash 9 8))
          :type (simple-array (unsigned-byte 16) (256)))
  (valoffset (make-array 18 :element-type 'fixnum :initial-element 0)
             :type (simple-array fixnum (18))))

;;; This structure contains all neccessary information about the decoded image
(defstruct descriptor
  (restart-interval 0 :type fixnum)
  (width 0 :type fixnum)
  (height 0 :type fixnum)
  buffer
  (qtables
   (make-array 4 :initial-contents
               (loop for j fixnum from 0 to 3
                     collecting
                        (make-array 8 :initial-contents
                                    (loop for i fixnum from 0 to 7
                                          collecting (make-array 8 :element-type 'uint8))))) :type (simple-array uint8-2d-array (*)))
  (huff-ac (make-array 2 :initial-contents
                       (list (make-huffstruct) (make-huffstruct))) :type (simple-array t (*)))
  (huff-dc (make-array 2 :initial-contents
                       (list (make-huffstruct) (make-huffstruct))) :type (simple-array t (*)))
  (cid (make-array 4) :type (simple-array t (*)))
  (scans (make-array 4 :initial-contents
                     (loop for i fixnum from 0 to 3 collecting (make-scan))) :type (simple-array t (*)))
  (H (make-array 4) :type (simple-array t (*)))
  (V (make-array 4) :type (simple-array t (*)))
  (iH (make-array 4) :type (simple-array t (*)))
  (iV (make-array 4) :type (simple-array t (*)))
  (qdest (make-array 4) :type (simple-array t (*)))
  (zz (make-array 64 :element-type 'sint16) :type sint16-array)
  (ncomp 0 :type fixnum)
  (ws (2d-sint16-array
       '(0  0  0  0  0  0  0  0)
       '(0  0  0  0  0  0  0  0)
       '(0  0  0  0  0  0  0  0)
       '(0  0  0  0  0  0  0  0)
       '(0  0  0  0  0  0  0  0)
       '(0  0  0  0  0  0  0  0)
       '(0  0  0  0  0  0  0  0)
       '(0  0  0  0  0  0  0  0))
      :type sint16-2d-array)	; Temporary workspace for IDCT
  (byte-reader #'(lambda () ;in case setup fails somehow
		   (error 'jpeg-decoder-error)) :type function)
  (source-cache)
  (bitreader nil)
  (adobe-app14-transform nil)
  (progressive-p nil)       ; t when decoding SOF2 (progressive) frame
  (coef-buffers nil)        ; vector of per-component fixnum arrays for DCT coefficients
  (blocks-per-row nil)      ; vector of per-component block-row counts
  (blocks-per-col nil)      ; vector of per-component block-column counts
  (y-plane nil :type (or null uint8-array))
  (cb-plane nil :type (or null uint8-array))
  (cr-plane nil :type (or null uint8-array))
  (planar-p nil :type boolean)
  (colorspace-conversion-p t :type boolean))

(defun allocate-coef-buffers (image Hmax Vmax)
  "Allocate per-component coefficient buffers for progressive decoding."
  (let* ((ncomp (descriptor-ncomp image))
         (width (descriptor-width image))
         (height (descriptor-height image))
         (mcus-x (ceiling width (* Hmax 8)))
         (mcus-y (ceiling height (* Vmax 8)))
         (buffers (make-array ncomp))
         (bpr (make-array ncomp))
         (bpc (make-array ncomp)))
    (loop for c fixnum from 0 below ncomp
          for h fixnum = (aref (descriptor-H image) c)
          for v fixnum = (aref (descriptor-V image) c)
          for blocks-x fixnum = (* mcus-x h)
          for blocks-y fixnum = (* mcus-y v)
          do (setf (aref bpr c) blocks-x)
             (setf (aref bpc c) blocks-y)
             (setf (aref buffers c)
                   (make-array (* blocks-x blocks-y 64)
                               :element-type 'fixnum
                               :initial-element 0)))
    (setf (descriptor-coef-buffers image) buffers)
    (setf (descriptor-blocks-per-row image) bpr)
    (setf (descriptor-blocks-per-col image) bpc)))

(defun read-jpeg-byte (image)
  (declare #.*optimize*)
  (the uint8 (funcall (descriptor-byte-reader image))))

;;; Reads an JPEG marker from the stream
(defun read-marker (s)
  (loop for b fixnum = (read-jpeg-byte s)
        when (/= b #xff) do (return b)))

;;; Reads 16-bit word from the stream
(defun read-word (s)
  "Reads 16-bit word from the stream"
  (let* ((msb (ash (read-jpeg-byte s) 8))
         (lsb (read-jpeg-byte s))
         (word (logior msb lsb)))
    word))

(eval-when (:compile-toplevel :load-toplevel :execute)
;;; APPn marker reading: just skipping the whole marker
  (defun read-app (s)
    "APPn marker reading: just skipping the whole marker"
    (loop for i fixnum from 0 below (minus (read-word s) 2) do
         (read-jpeg-byte s)))

;;; COM marker reading, same as read-app
  (setf (symbol-function 'read-com) #'read-app))

;;; Sets up restart interval
(defun read-dri (image)
  "Reads restart interval"
  (read-jpeg-byte image) ; skipping length
  (read-jpeg-byte image)
  (setf (descriptor-restart-interval image) (read-word image)))

;;; 'Inverse zigzag transform'
(defun izigzag (inbuf zzbuf)
  (declare #.*optimize*
           (type sint16-array inbuf)
	   (type sint16-2d-array zzbuf)
	   (type uint8-2d-array +zigzag-index+))
  "Performs inverse zigzag block arrangement"
  (loop for zrow of-type uint8-array across +zigzag-index+
        for row of-type sint16-array across zzbuf do
        (loop for pos fixnum across zrow
              for x fixnum from 0 do
              (setf (aref row x) (aref inbuf pos))))
  zzbuf)

(defun izigzag8 (inbuf zzbuf)
  (declare #.*optimize*
           (type uint8-array inbuf)
	   (type uint8-2d-array zzbuf +zigzag-index+))
  "Performs inverse zigzag block arrangement"
  (loop for zrow of-type uint8-array across +zigzag-index+
        for row of-type uint8-array across zzbuf do
        (loop for pos fixnum across zrow
              for x fixnum from 0 do
              (setf (aref row x) (aref inbuf pos))))
  zzbuf)

;;; Reads in quantization tables
(defun read-dqt (image)
  "Reads in quantization tables"
  (let ((len (minus (read-word image) 2)))
    (loop for i fixnum from (1- len) downto 0 by 65
          for tq fixnum = (logand (read-jpeg-byte image) 7)
          with intable of-type uint8-array = (make-array 64 :element-type 'uint8)
          for table = (aref (descriptor-qtables image) tq) do
          (loop for pos from 0 to 63 do
                (setf (aref intable pos) (read-jpeg-byte image)))
          (izigzag8 intable table))))

;;; Builds up decoder tables
(defun build-decoder-tables (bits huffcode &optional tables)
  "Builds up decoder-specific tables, including fast lookahead and valoffset."
  (let ((maxcode (make-array 18 :element-type 'fixnum :initial-element -1))
        (mincode (make-array 17 :element-type 'fixnum :initial-element 0))
        (valptr (make-array 17 :element-type 'fixnum :initial-element 0)))
    (declare (type (simple-array fixnum (18)) maxcode)
             (type (simple-array fixnum (17)) mincode valptr)
             (type (simple-array uint8 (*)) bits)
             (type (simple-array fixnum (*)) huffcode))
    (loop with i fixnum = 0
          and j fixnum = 0 do
          (when (loop
                 (incf i)
                 (when (> i 16)
                   (return t))
                 (cond ((zerop (aref bits (1- i)))
                        (setf (aref maxcode i) -1))
                       (t (return nil))))
            (return))
          (setf (aref valptr i) j)
          (setf (aref mincode i) (aref huffcode j))
          (incf j (1- (aref bits (1- i))))
          (setf (aref maxcode i) (aref huffcode j))
          (incf j))
    (setf (aref maxcode 17) #xFFFFFF)
    (when tables
      (let ((valoffset (huffstruct-valoffset tables))
            (lookup (huffstruct-lookup tables))
            (huffval (huffstruct-huffval tables)))
        (declare (type (simple-array fixnum (18)) valoffset)
                 (type (simple-array (unsigned-byte 16) (256)) lookup)
                 (type (simple-array uint8 (*)) huffval))
        ;; Compute valoffset for fast symbol lookup: valoffset[l] = valptr[l] - mincode[l]
        (loop for l from 1 to 16 do
          (if (/= (aref maxcode l) -1)
              (setf (aref valoffset l) (- (aref valptr l) (aref mincode l)))
              (setf (aref valoffset l) 0)))
        (setf (aref valoffset 17) 0)
        ;; Populate 256-entry lookahead table (HUFF_LOOKAHEAD = 8)
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
                (incf p)))))))
    (values maxcode mincode valptr)))

;;; Loads huffman tables
(defun read-dht (image)
  "Loads huffman tables on specified destinations"
  (let ((len (minus (read-word image) 2))
        (count 0))
    (loop for tcth fixnum = (read-jpeg-byte image)
          for tc fixnum = (ash tcth -4)
          for th fixnum = (logand tcth 15)
          for tables = (if (zerop tc)
                           (aref (descriptor-huff-ac image) th)
                         (aref (descriptor-huff-dc image) th))
          for bits = (huffstruct-bits tables)
          for sum fixnum = 0 do
          (loop for i fixnum from 0 to 15
                for entry fixnum = (read-jpeg-byte image) do
                (incf sum entry)
                (setf (aref bits i) entry))
          (setf (huffstruct-huffval tables)
                (make-array sum :element-type 'uint8
			    :initial-contents (loop for i fixnum from 0 below sum
                                                        collecting (read-jpeg-byte image))))
          (incf count (plus sum 17))
	 (multiple-value-bind (maxcode mincode valptr)
              (build-decoder-tables bits (setf (huffstruct-huffcode tables)
                                               (build-universal-tables bits))
                                   tables)
            ;;(declare (type uint8-array mincode valptr))
            (setf (huffstruct-maxcode tables) maxcode)
            (setf (huffstruct-mincode tables) mincode)
            (setf (huffstruct-valptr tables) valptr))
          (unless (< count len) (return t)))))

;;; APP14 is assumed to be Adobe proprietary extension marker
;;; We parse it to figure out if some special color transform is necessary
(defun read-app14 (image)
  (let ((length (minus (read-word image) 2)))
    (loop repeat 11 do (read-jpeg-byte image))
    (setf (descriptor-adobe-app14-transform image)
	  (case (read-jpeg-byte image)
	    (0 :unknown)
	    (1 :ycbcr-rgb)
	    (2 :ycck-cmyk)
	    (otherwise :invalid)))
    (loop repeat (- length 12) do (read-jpeg-byte image))))

;;; Reads tables etc., returns the first unrecognized marker it met
(defun interpret-markers (image term)
  "Reads tables etc., returns the first unrecognized marker it met"
  (loop for mk fixnum = (cond ((zerop term) (read-marker image))
                              (t term)) do
       (setf term 0)
       (cond ((= mk +M_APP14+) (read-app14 image)) ;Adobe marker
	     ((= #xe0 (logand #xf0 mk)) ; Unrecognized APPn marker
	      (read-app image))
	     (t (cond ((= mk +M_DAC+) (error 'unsupported-arithmetic-encoding))
		      ((= mk +M_DRI+) (read-dri image))
		      ((= mk +M_DHT+) (read-dht image))
		      ((= mk +M_DQT+) (read-dqt image))
		      ((= mk +M_COM+) (read-com image))		      
		      (t (return mk)))))))

;;; ---------------------------------------------------------------------------
;;; Fast Huffman Entropy Decoder (jdhuff.c architecture)
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
(defun fast-decode-symbol (hs br image)
  "Decodes one Huffman symbol using 8-bit lookup table with slow-path fallback."
  (declare (type huffstruct hs)
           (type bitreader br)
           (optimize (speed 3) (safety 0)))
  (when (< (bitreader-bits-left br) 8)
    (refill-bit-buffer br image))
  
  (let* ((left (bitreader-bits-left br))
         (buf (bitreader-buffer br))
         (lookup (huffstruct-lookup hs)))
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
                            (maxcode (huffstruct-maxcode hs))
                            (valoffset (huffstruct-valoffset hs))
                            (huffval (huffstruct-huffval hs)))
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
                            (maxcode (huffstruct-maxcode hs))
                            (valoffset (huffstruct-valoffset hs))
                            (huffval (huffstruct-huffval hs)))
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
              (maxcode (huffstruct-maxcode hs))
              (valoffset (huffstruct-valoffset hs))
              (huffval (huffstruct-huffval hs)))
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

(declaim (inline fast-decode-block))
(defun fast-decode-block (zz dc-tab ac-tab br image)
  (declare (type sint16-array zz)
           (type huffstruct dc-tab ac-tab)
           (type bitreader br)
           (optimize (speed 3) (safety 0)))
  ;; 1. Decode DC
  (let ((tt (fast-decode-symbol dc-tab br image)))
    (declare (type fixnum tt))
    (setf (aref zz 0)
          (if (zerop tt)
              0
              (fast-extend (fast-get-bits tt br image) tt))))
  
  ;; 2. Clear AC coefficients
  (fill zz 0 :start 1)
  
  ;; 3. Decode AC coefficients
  (loop with k fixnum = 1
        for rs fixnum = (fast-decode-symbol ac-tab br image)
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

;;; EXTEND procedure, as described in the standard
(defun extend (v tt)
  "EXTEND procedure, as described in spec."
  (let ((vt (if (> tt 0)
                (ash 1 (the (and fixnum (integer 0)) (minus tt 1)))
                0)))
    (declare (type fixnum v vt tt)
             #.*optimize*)
    (if (< v vt)
        (plus v (plus 1 (ash -1 (the (and fixnum (integer 0)) tt))))
      v)))

;;; Returns the closure which reads specified numbers of bits from the stream
(defun make-nextbit (b cnt)
  "Returns the closure which reads specified numbers of bits from the stream"
  #'(lambda (s)
      (let ((bit 0))
        (declare #.*optimize*
                 (type fixnum b cnt bit))
        (when (zerop cnt)
          (setf b (read-jpeg-byte s))
          (setf cnt 8)
          (when (= b #xff)
            (let ((b2 (read-jpeg-byte s)))
              (declare (type uint8 b2))
              (cond ((zerop b2))
                    ((<= +M_RST0+ b2 +M_RST7+)
                     (throw 'marker 'restart))
                    ((= b2 +M_DNL+)
                     (error 'unsupported-dnl-marker))
                    (t (throw 'marker b2))))))
        (decf cnt)
        (setf bit (ash (logand b 255) -7))
        (setf b (ash b 1))
        bit)))

;;; The DECODE procedure
(defun decode (maxcode mincode valptr huffval nextbit s)
  "The DECODE procedure, as described in CCITT rec."
  (let ((i 1)
        (code (funcall nextbit s)))
    (declare #.*optimize*
             (type (simple-array uint8 (*)) huffval)
             (type (simple-array fixnum (*)) maxcode mincode valptr)
             (type fixnum i code)
             (type function nextbit))
    (loop while (> code (aref maxcode i)) do
          (incf i)
          (setf code (plus (ash code 1) (funcall nextbit s))))
    (aref huffval (plus (aref valptr i) (minus code (aref mincode i))))))

;;; Recieves ssss bits from the stream
(defun recieve (ssss nextbit s)
  "Recieves ssss bits from the stream"
  (let ((v 0))
    (declare #.*optimize*
             (type fixnum v ssss)
             (type function nextbit))
    (loop for i fixnum from 0
          until (= i ssss) do
          (setf v (plus (ash v 1) (funcall nextbit s))))
    v))

;;; Decodes AC coefficients
(defun decode-ac (zz maxcode mincode valptr huffval nextbit s)
  "Decodes AC coefficients"
  (declare #.*optimize*
           (type (simple-array uint8 (*)) huffval))
  (fill zz 0 :start 1)
  (loop with k fixnum = 1
        for rs fixnum = (decode maxcode mincode valptr huffval nextbit s)
        for ssss fixnum = (logand rs 15)
        for r fixnum = (ash rs -4) do
        (cond ((zerop ssss)
               (if (= r 15)
                   (incf k 16)
                 (return zz)))
              (t (incf k r)
                 (setf (aref zz k)
                       (extend (recieve ssss nextbit s) ssss))
                 (if (= k 63)
                     (return zz)
                   (incf k))))))

;;; Decodes DC value
(defun decode-dc (maxcode mincode valptr huffval nextbit s)
  "Decodes DC value"
  (let ((tt (decode maxcode mincode valptr huffval nextbit s)))
  (declare #.*optimize*
           (type (simple-array uint8 (*)) huffval)
           (fixnum tt))
    (extend (recieve tt nextbit s) tt)))

;;; Decodes single 8x8 block
(defun decode-block (zz tabs nextbit s)
  "Reads one 8x8 block. Doesn't deals with predictors."
  (let ((tdc (aref tabs 0))
        (tac (aref tabs 1)))
  (declare #.*optimize*
           (type sint16-array zz)
	   (type (simple-array sint16-2d-array (*)) tabs)
           (type huffstruct tac tdc))
  (setf (aref zz 0) (decode-dc (huffstruct-maxcode tdc)
                                (huffstruct-mincode tdc)
                                (huffstruct-valptr tdc)
                                (huffstruct-huffval tdc) nextbit s))
  (decode-ac zz
             (huffstruct-maxcode tac)
             (huffstruct-mincode tac)
             (huffstruct-valptr tac)
             (huffstruct-huffval tac) nextbit s)
  zz))

;;; Dequanitzation
(defun dequantize (x y block table)
  "Dequantizes a single sample"
  (declare #.*optimize*
           (type fixnum x y))
  (mul (s16ref block x y) (u8ref table x y)))

;;;Macro that bounds value in IDCT
(defmacro dct-limit (n)
  `(aref *idct-limit-array* (logand (plus ,n 255) 511)))

;;; Inverse LLM DCT and dequantization (scalar reference)
(defun inverse-llm-dct-scalar (block q-table ws)
  "Performs scalar Inverse LLM DCT and dequantization"
  (let ((tmp0 0) (tmp1 0) (tmp2 0) (tmp3 0)
        (tmp10 0) (tmp11 0) (tmp12 0) (tmp13 0)
        (z1 0) (z2 0) (z3 0) (z4 0) (z5 0)
        (dcval 0))
    (declare #.*optimize*
             (type fixnum tmp0 tmp1 tmp2 tmp3 tmp10 tmp11 tmp12 tmp13 z1 z2 z3 z4 z5 dcval)
             (type sint16-2d-array block ws)
             (type uint8-2d-array q-table)
             (dynamic-extent tmp0 tmp1 tmp2 tmp3 tmp10 tmp11 tmp12 tmp13 z1 z2 z3 z4 z5 dcval))
    (loop for dptr fixnum from 0 to 7 ; iterating over columns
          if (and (zerop (s16ref block dptr 1))
                  (zerop (s16ref block dptr 2))
                  (zerop (s16ref block dptr 3))
                  (zerop (s16ref block dptr 4))
                  (zerop (s16ref block dptr 5))
                  (zerop (s16ref block dptr 6))
                  (zerop (s16ref block dptr 7))) do
          (setf dcval (ash (the fixnum (dequantize dptr 0 block q-table)) 1))
          (setf (s16ref ws dptr 0) dcval)
          (setf (s16ref ws dptr 1) dcval)
          (setf (s16ref ws dptr 2) dcval)
          (setf (s16ref ws dptr 3) dcval)
          (setf (s16ref ws dptr 4) dcval)
          (setf (s16ref ws dptr 5) dcval)
          (setf (s16ref ws dptr 6) dcval)
          (setf (s16ref ws dptr 7) dcval)
          else do
          (setf z2 (dequantize dptr 2 block q-table))
          (setf z3 (dequantize dptr 6 block q-table))
          (setf z1 (mul (plus z2 z3) +FIX-0-541196100+))
          (setf tmp2 (plus z1 (mul z3 (- +FIX-1-847759065+))))
          (setf tmp3 (plus z1 (mul z2 +FIX-0-765366865+)))
          (setf z2 (dequantize dptr 0 block q-table))
          (setf z3 (dequantize dptr 4 block q-table))
          (setf tmp0 (ash (plus z2 z3) dct-shift))
          (setf tmp1 (ash (minus z2 z3) dct-shift))
          (setf tmp10 (plus tmp0 tmp3))
          (setf tmp13 (minus tmp0 tmp3))
          (setf tmp11 (plus tmp1 tmp2))
          (setf tmp12 (minus tmp1 tmp2))
          (setf tmp0 (dequantize dptr 7 block q-table))
          (setf tmp1 (dequantize dptr 5 block q-table))
          (setf tmp2 (dequantize dptr 3 block q-table))
          (setf tmp3 (dequantize dptr 1 block q-table))
          (setf z1 (plus tmp0 tmp3))
          (setf z2 (plus tmp1 tmp2))
          (setf z3 (plus tmp0 tmp2))
          (setf z4 (plus tmp1 tmp3))
          (setf z5 (mul (plus z3 z4) +FIX-1-175875602+))
          (setf tmp0 (mul tmp0 +FIX-0-298631336+))
          (setf tmp1 (mul tmp1 +FIX-2-053119869+))
          (setf tmp2 (mul tmp2 +FIX-3-072711026+))
          (setf tmp3 (mul tmp3 +FIX-1-501321110+))
          (setf z1 (mul z1 (- +FIX-0-899976223+)))
          (setf z2 (mul z2 (- +FIX-2-562915447+)))
          (setf z3 (mul z3 (- +FIX-1-961570560+)))
          (setf z4 (mul z4 (- +FIX-0-390180644+)))
          (incf z3 z5)
          (incf z4 z5)
          (incf tmp0 (plus z1 z3))
          (incf tmp1 (plus z2 z4))
          (incf tmp2 (plus z2 z3))
          (incf tmp3 (plus z1 z4))
          (setf (s16ref ws dptr 0) (descale (plus tmp10 tmp3) +shift-1+))
          (setf (s16ref ws dptr 7) (descale (minus tmp10 tmp3) +shift-1+))
          (setf (s16ref ws dptr 1) (descale (plus tmp11 tmp2) +shift-1+))
          (setf (s16ref ws dptr 6) (descale (minus tmp11 tmp2) +shift-1+))
          (setf (s16ref ws dptr 2) (descale (plus tmp12 tmp1) +shift-1+))
          (setf (s16ref ws dptr 5) (descale (minus tmp12 tmp1) +shift-1+))
          (setf (s16ref ws dptr 3) (descale (plus tmp13 tmp0) +shift-1+))
          (setf (s16ref ws dptr 4) (descale (minus tmp13 tmp0) +shift-1+)))

    (loop for row of-type sint16-array across block ; iterating over rows
          for inrow of-type sint16-array across ws
          if (not (find-if-not #'zerop inrow :start 1)) do
          (setf dcval (dct-limit (descale (aref inrow 0) 4)))
          (setf (aref row 0) dcval)
          (setf (aref row 1) dcval)
          (setf (aref row 2) dcval)
          (setf (aref row 3) dcval)
          (setf (aref row 4) dcval)
          (setf (aref row 5) dcval)
          (setf (aref row 6) dcval)
          (setf (aref row 7) dcval)
          else do
          (setf z2 (aref inrow 2))
          (setf z3 (aref inrow 6))
          (setf z1 (mul (plus z2 z3) +FIX-0-541196100+))
          (setf tmp2 (plus z1 (mul z3 (- +FIX-1-847759065+))))
          (setf tmp3 (plus z1 (mul z2 +FIX-0-765366865+)))
          (setf tmp0 (ash (plus (aref inrow 0) (aref inrow 4)) dct-shift))
          (setf tmp1 (ash (minus (aref inrow 0) (aref inrow 4)) dct-shift))
          (setf tmp10 (plus tmp0 tmp3))
          (setf tmp13 (minus tmp0 tmp3))
          (setf tmp11 (plus tmp1 tmp2))
          (setf tmp12 (minus tmp1 tmp2))
          (setf tmp0 (aref inrow 7))
          (setf tmp1 (aref inrow 5))
          (setf tmp2 (aref inrow 3))
          (setf tmp3 (aref inrow 1))
          (setf z1 (plus tmp0 tmp3))
          (setf z2 (plus tmp1 tmp2))
          (setf z3 (plus tmp0 tmp2))
          (setf z4 (plus tmp1 tmp3))
          (setf z5 (mul (plus z3 z4) +FIX-1-175875602+))
          (setf tmp0 (mul tmp0 +FIX-0-298631336+))
          (setf tmp1 (mul tmp1 +FIX-2-053119869+))
          (setf tmp2 (mul tmp2 +FIX-3-072711026+))
          (setf tmp3 (mul tmp3 +FIX-1-501321110+))
          (setf z1 (mul z1 (- +FIX-0-899976223+)))
          (setf z2 (mul z2 (- +FIX-2-562915447+)))
          (setf z3 (mul z3 (- +FIX-1-961570560+)))
          (setf z4 (mul z4 (- +FIX-0-390180644+)))
          (incf z3 z5)
          (incf z4 z5)
          (incf tmp0 (plus z1 z3))
          (incf tmp1 (plus z2 z4))
          (incf tmp2 (plus z2 z3))
          (incf tmp3 (plus z1 z4))
          (setf (aref row 0) (dct-limit (descale (plus tmp10 tmp3) +shift+4+)))
          (setf (aref row 7) (dct-limit (descale (minus tmp10 tmp3) +shift+4+)))
          (setf (aref row 1) (dct-limit (descale (plus tmp11 tmp2) +shift+4+)))
          (setf (aref row 6) (dct-limit (descale (minus tmp11 tmp2) +shift+4+)))
          (setf (aref row 2) (dct-limit (descale (plus tmp12 tmp1) +shift+4+)))
          (setf (aref row 5) (dct-limit (descale (minus tmp12 tmp1) +shift+4+)))
          (setf (aref row 3) (dct-limit (descale (plus tmp13 tmp0) +shift+4+)))
          (setf (aref row 4) (dct-limit (descale (minus tmp13 tmp0) +shift+4+))))))

(declaim (inline inverse-llm-dct))
(defun inverse-llm-dct (block q-table ws)
  "Performs Inverse LLM DCT and dequantization, accelerating via SIMD when available."
  (if (and (not (eql *use-simd* :off)) *inverse-llm-dct-impl*)
      (funcall (the function *inverse-llm-dct-impl*) block q-table ws)
      (inverse-llm-dct-scalar block q-table ws)))

;;; Places decoded block into the image buffer, with necessary upsampling
(declaim (inline upsample-1x1-fast upsample-2x2-fast))

(defun upsample-1x1-fast (buffer block offset base-pos nwidth)
  (declare (type uint8-array buffer)
           (type sint16-2d-array block)
           (type fixnum offset base-pos nwidth)
           (optimize (speed 3) (safety 0)))
  (let ((pos (+ base-pos offset)))
    (declare (type fixnum pos))
    (dotimes (r 8)
      (let ((row (aref block r)))
        (declare (type sint16-array row))
        (setf (aref buffer (+ pos 0))  (aref row 0)
              (aref buffer (+ pos 3))  (aref row 1)
              (aref buffer (+ pos 6))  (aref row 2)
              (aref buffer (+ pos 9))  (aref row 3)
              (aref buffer (+ pos 12)) (aref row 4)
              (aref buffer (+ pos 15)) (aref row 5)
              (aref buffer (+ pos 18)) (aref row 6)
              (aref buffer (+ pos 21)) (aref row 7)))
      (incf pos nwidth))))

(defun upsample-2x2-fast (buffer block offset base-pos nwidth)
  (declare (type uint8-array buffer)
           (type sint16-2d-array block)
           (type fixnum offset base-pos nwidth)
           (optimize (speed 3) (safety 0)))
  (let ((row-pos (+ base-pos offset))
        (nwidth2 (* nwidth 2)))
    (declare (type fixnum row-pos nwidth2))
    (dotimes (r 8)
      (let ((row (aref block r))
            (p0 row-pos)
            (p1 (+ row-pos nwidth)))
        (declare (type sint16-array row)
                 (type fixnum p0 p1))
        (macrolet ((put2x2 (idx col)
                     `(let ((v (aref row ,col)))
                        (setf (aref buffer (+ p0 ,idx))       v
                              (aref buffer (+ p0 ,(+ idx 3))) v
                              (aref buffer (+ p1 ,idx))       v
                              (aref buffer (+ p1 ,(+ idx 3))) v))))
          (put2x2 0 0)
          (put2x2 6 1)
          (put2x2 12 2)
          (put2x2 18 3)
          (put2x2 24 4)
          (put2x2 30 5)
          (put2x2 36 6)
          (put2x2 42 7)))
      (incf row-pos nwidth2))))

;;; COMP-INDEX DOCUMENTATION:
;;; Planar decoder path: index 0=Y, 1=Cb, 2=Cr, selecting a plane.
;;; Interleaved output byte order: byte 0=B, 1=G, 2=R.
;;; The two indices coincide numerically only because the planar components and the
;;; interleaved output bytes happen to be ordered the same way in the standard 3-component case.
;;; Do not rely on this coincidence for 4-component (YCCK/CMYK) or grayscale paths.

(defun upsample-default (image scan block x y H V comp-index nwidth nw nx dend)
  "Places decoded block into the image buffer, with necessary upsampling.
COMP-INDEX indicates the 0-based component index in (descriptor-cid image)."
  (declare #.*optimize*
           (type sint16-2d-array block)
           (type fixnum x y H V comp-index nwidth nw nx dend))
  (let* ((buffer (descriptor-buffer image))
         (xbase (+ (scan-x scan) x))
         (ybase (+ (scan-y scan) y))
         (img-width (descriptor-width image))
         (img-height (descriptor-height image))
         (ncomp (descriptor-ncomp image))
         (base-pos (+ (* ybase nwidth) (* xbase ncomp))))
    (declare (type uint8-array buffer)
             (type fixnum xbase ybase img-width img-height ncomp base-pos))
    (cond
      ;; 1. Fast path: H=1, V=1, interior block (Y in 4:2:0 or 4:4:4)
      ((and (= H 1) (= V 1) (= ncomp 3) (<= (+ xbase 8) img-width) (<= (+ ybase 8) img-height))
       (upsample-1x1-fast buffer block comp-index base-pos nwidth))
      
      ;; 2. Fast path: H=2, V=2, interior block (Cb/Cr in 4:2:0)
      ((and (= H 2) (= V 2) (= ncomp 3) (<= (+ xbase 16) img-width) (<= (+ ybase 16) img-height))
       (upsample-2x2-fast buffer block comp-index base-pos nwidth))
      
      ;; 3. General fallback for boundary blocks or non-standard sampling
      (t
       (let* ((nxbase (* xbase ncomp))
              (nybase (* ybase nwidth)))
         (declare (type fixnum nxbase nybase))
         (loop for row of-type sint16-array across block
               for y-idx fixnum from ybase below (descriptor-height image) by V
               for ypos fixnum from nybase by nw do
               (loop for val fixnum across row
                     for x-idx fixnum from xbase below (descriptor-width image) by H
                     for pos fixnum from (the fixnum (+ ypos comp-index nxbase)) by nx do
                     (if (= 1 H V)
                         (setf (aref buffer pos) val)
                         (loop for dy fixnum from 0 below V
                               for dypos fixnum from pos below dend by nwidth
                               for dxend from (* (+ (1+ y-idx) dy) nwidth) by nwidth do
                               (loop for dx fixnum from 0 below H
                                     for dpos fixnum from dypos below dxend by 3 do
                                     (setf (aref buffer dpos) val)))))))))))

(declaim (inline upsample))
(defun upsample (image scan block x y H V comp-index nwidth nw nx dend)
  "Places decoded block into the image buffer or planar storage, with necessary upsampling.
COMP-INDEX is the 0-based component index in (descriptor-cid image):
  0 -> Y, 1 -> Cb, 2 -> Cr."
  (declare #.*optimize*
           (type sint16-2d-array block)
           (type fixnum x y H V comp-index nwidth nw nx dend))
  (if (and (not (eql *use-simd* :off)) *upsample-impl* (descriptor-planar-p image))
      (funcall (the function *upsample-impl*) image scan block x y H V comp-index nwidth nw nx dend)
      (upsample-default image scan block x y H V comp-index nwidth nw nx dend)))

;;; Reads and decodes either whole scan or restart interval
(defun decode-chunk (image scan zzbuf)
  "Reads and decodes either a whole scan (if no restarts) or restart interval"
  (let* ((br (or (descriptor-bitreader image)
                 (setf (descriptor-bitreader image) (make-bitreader))))
         (ncomp (scan-ncomp scan))
         (nwidth (mul (descriptor-width image) (descriptor-ncomp image)))
         (dend (mul (descriptor-height image) nwidth))
         (fr (make-array ncomp :initial-contents
                         (loop
                           ;; collecting sampling rates for a components in the scan
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
                            (loop for i fixnum from 0 below ncomp ; collecting sampling frequencies
                                  for cid fixnum = (first (aref (scan-cdesc scan) i))
                                  for pos fixnum = (position cid (descriptor-cid image))
                                  collecting (list (aref (descriptor-iH image) pos)
                                                   (aref (descriptor-iV image) pos)))))
         (preds (make-array ncomp :initial-element 0 :element-type 'sint16))
         (dc-tables (make-array ncomp
                      :initial-contents
                      (loop for i fixnum from 0 below ncomp
                            for td fixnum = (ash (the fixnum (second (aref (scan-cdesc scan) i))) -4)
                            collecting (aref (descriptor-huff-ac image) td))))
         (ac-tables (make-array ncomp
                      :initial-contents
                      (loop for i fixnum from 0 below ncomp
                            for ta fixnum = (logand (second (aref (scan-cdesc scan) i)) 15)
                            collecting (aref (descriptor-huff-dc image) ta)))))
    (declare #.*optimize*
             (type fixnum ncomp Hmax Vmax x-growth y-growth nwidth)
             (type (simple-array t (*)) freqs fr dc-tables ac-tables)
             (type (simple-array sint16 (*)) preds)
             (dynamic-extent fr freqs))
    (setf (bitreader-buffer br) 0
          (bitreader-bits-left br) 0
          (bitreader-unread-marker br) nil)
    (catch 'marker
      (loop
        do (loop for comp fixnum from 0 below ncomp
                 ;; 0-based component index in (descriptor-cid image):
                 ;; 0 -> Y, 1 -> Cb, 2 -> Cr.
                 for comp-index fixnum =
                    (position (the fixnum (first (aref (scan-cdesc scan) comp)))
                              (descriptor-cid image))
                 for q-tab = (aref (descriptor-qtables image) (aref (descriptor-qdest image) comp))
                 for H fixnum = (first (aref freqs comp))
                 for V fixnum = (second (aref freqs comp))
                 for nw fixnum = (mul nwidth V)
                 for nx fixnum = (mul (descriptor-ncomp image) H)
                 for blocks-y fixnum = (second (aref fr comp))
                 for blocks-x fixnum = (first (aref fr comp))
                 for dc-tab = (aref dc-tables comp)
                 for ac-tab = (aref ac-tables comp) do
                    (loop for y fixnum from 0 below blocks-y
                          for y-pos fixnum from (mul (ash y 3) V) by (ash V 3) do
                            (loop for x fixnum from 0 below blocks-x
                                  for x-pos fixnum from (mul (ash x 3) H) by (ash H 3)
                                  for decoded-block of-type sint16-2d-array =
                                    (izigzag (fast-decode-block (descriptor-zz image)
                                                                dc-tab ac-tab br image) zzbuf) do
                                    ;; DC decoding and predictor update
                                    (incf (s16ref decoded-block 0 0) (aref preds comp))
                                    (setf (aref preds comp) (s16ref decoded-block 0 0))
                                    (when (and (< (plus x-pos (scan-x scan)) (descriptor-width image))
                                               (< (plus y-pos (scan-y scan)) (descriptor-height image)))
                                      ;; inverse DCT and block write to the buffer
                                      (inverse-llm-dct decoded-block q-tab (descriptor-ws image))
                                      (upsample image scan decoded-block x-pos y-pos
                                                H V comp-index nwidth nw nx dend)))))
           (incf (scan-x scan) x-growth)
           (when (<= (descriptor-width image) (scan-x scan))
             (incf (scan-y scan) y-growth)
             (setf (scan-x scan) 0))
           (when (>= (scan-y scan) (descriptor-height image))
             (if (bitreader-unread-marker br)
                 (throw 'marker (bitreader-unread-marker br))
                 (loop for b fixnum = (br-read-byte br image)
                       when (= b #xFF) do
                         (loop for b2 fixnum = (br-read-byte br image)
                               while (= b2 #xFF)
                               finally
                                 (unless (zerop b2)
                                   (throw 'marker b2))))))
           (when (eq (bitreader-unread-marker br) 'restart)
             (throw 'marker 'restart))))))

;;; ============================================================
;;; Progressive JPEG decoding functions
;;; ============================================================

;;; Decodes DC coefficient for first DC scan (Ss=0, Se=0, Ah=0)
(defun decode-dc-first (coef-buf offset al dc-table nextbit image preds comp)
  "Decode DC coefficient for first scan. Stores ash(value, al) into coef buffer."
  (declare #.*optimize*
           (type fixnum offset al comp)
           (type (simple-array fixnum (*)) coef-buf)
           (type (simple-array sint16 (*)) preds)
           (type huffstruct dc-table)
           (type function nextbit))
  (let* ((tt (decode (huffstruct-maxcode dc-table)
                     (huffstruct-mincode dc-table)
                     (huffstruct-valptr dc-table)
                     (huffstruct-huffval dc-table) nextbit image))
         (diff (extend (recieve tt nextbit image) tt)))
    (declare (type fixnum tt diff))
    (incf diff (aref preds comp))
    (setf (aref preds comp) diff)
    (setf (aref coef-buf offset) (ash diff al))))

;;; Decodes DC refinement scan (Ss=0, Se=0, Ah>0) - one raw bit per block
(defun decode-dc-refine (coef-buf offset al nextbit image)
  "Refine DC coefficient: read one bit and OR into position al."
  (declare #.*optimize*
           (type fixnum offset al)
           (type (simple-array fixnum (*)) coef-buf)
           (type function nextbit))
  (let ((bit (funcall nextbit image)))
    (declare (type fixnum bit))
    (when (= bit 1)
      (setf (aref coef-buf offset)
            (logior (aref coef-buf offset) (ash 1 al))))))

;;; Decodes AC coefficients for first AC scan (Ss>0, Ah=0) with EOBRUN
(defun decode-ac-first (coef-buf offset ss se al ac-table nextbit image eobrun)
  "Decode AC first scan for coefficients ss..se with point transform al.
   Returns updated eobrun value."
  (declare #.*optimize*
           (type fixnum offset ss se al eobrun)
           (type (simple-array fixnum (*)) coef-buf)
           (type huffstruct ac-table)
           (type function nextbit))
  (when (> eobrun 0)
    (return-from decode-ac-first (1- eobrun)))
  (let ((k ss))
    (declare (type fixnum k))
    (loop while (<= k se) do
          (let* ((rs (decode (huffstruct-maxcode ac-table)
                             (huffstruct-mincode ac-table)
                             (huffstruct-valptr ac-table)
                             (huffstruct-huffval ac-table) nextbit image))
                 (s (logand rs 15))
                 (r (ash rs -4)))
            (declare (type fixnum rs s r))
            (cond
              ;; Non-zero coefficient
              ((> s 0)
               (incf k r)
               (setf (aref coef-buf (+ offset k))
                     (ash (extend (recieve s nextbit image) s) al))
               (incf k))
              ;; ZRL: skip 16 zeros
              ((= r 15)
               (incf k 16))
              ;; EOBn: end of band run
              (t
               (let ((run (ash 1 r)))
                 (declare (type fixnum run))
                 (when (> r 0)
                   (incf run (recieve r nextbit image)))
                 ;; Current block is the first in the run
                 (return-from decode-ac-first (1- run))))))))
  0)

;;; Decodes AC refinement scan (Ss>0, Ah>0)
(defun decode-ac-refine (coef-buf offset ss se al ac-table nextbit image eobrun)
  "Refine AC coefficients with correction bits and new coefficient introduction.
   Returns updated eobrun value."
  (declare #.*optimize*
           (type fixnum offset ss se al eobrun)
           (type (simple-array fixnum (*)) coef-buf)
           (type huffstruct ac-table)
           (type function nextbit))
  (let ((p1 (ash 1 al))
        (m1 (- (ash 1 al)))
        (k ss))
    (declare (type fixnum p1 m1 k))
    (flet ((apply-correction (pos)
             "Read one correction bit for a nonzero coefficient at coef-buf[pos]."
             (declare (type fixnum pos))
             (let ((coeff (aref coef-buf pos)))
               (declare (type fixnum coeff))
               (when (/= coeff 0)
                 (when (= (funcall nextbit image) 1)
                   (if (> coeff 0)
                       (when (zerop (logand coeff p1))
                         (incf (aref coef-buf pos) p1))
                       (when (zerop (logand coeff p1))
                         (incf (aref coef-buf pos) m1))))
                 t))))
      ;; If in an EOB run, just apply correction bits to existing nonzeros
      (when (> eobrun 0)
        (loop for j fixnum from ss to se do
              (apply-correction (+ offset j)))
        (return-from decode-ac-refine (1- eobrun)))
      ;; Decode new symbols
      (loop while (<= k se) do
            (let* ((rs (decode (huffstruct-maxcode ac-table)
                               (huffstruct-mincode ac-table)
                               (huffstruct-valptr ac-table)
                               (huffstruct-huffval ac-table) nextbit image))
                   (s (logand rs 15))
                   (r (ash rs -4)))
              (declare (type fixnum rs s r))
              (cond
                ;; s=1: new nonzero coefficient
                ((= s 1)
                 (let ((sign-bit (funcall nextbit image)))
                   (declare (type fixnum sign-bit))
                   (let ((new-value (if (= sign-bit 1) p1 m1)))
                     (declare (type fixnum new-value))
                     ;; Skip r zero-valued positions (applying corrections to nonzeros)
                     (loop while (and (<= k se) (> r 0)) do
                           (let ((coeff (aref coef-buf (+ offset k))))
                             (declare (type fixnum coeff))
                             (if (/= coeff 0)
                                 (apply-correction (+ offset k))
                                 (decf r)))
                           (incf k))
                     ;; Skip past remaining nonzeros to find the zero position
                     (loop while (and (<= k se)
                                      (/= (aref coef-buf (+ offset k)) 0))
                           do (apply-correction (+ offset k))
                              (incf k))
                     ;; Place the new coefficient
                     (when (<= k se)
                       (setf (aref coef-buf (+ offset k)) new-value)
                       (incf k)))))
                ;; s=0, r=15: ZRL with correction bits
                ((= r 15)
                 (let ((zeros-to-skip 16))
                   (declare (type fixnum zeros-to-skip))
                   (loop while (and (<= k se) (> zeros-to-skip 0)) do
                         (let ((coeff (aref coef-buf (+ offset k))))
                           (declare (type fixnum coeff))
                           (if (/= coeff 0)
                               (apply-correction (+ offset k))
                               (decf zeros-to-skip)))
                         (incf k))))
                ;; s=0, r<15: EOBn
                (t
                 (let ((run (ash 1 r)))
                   (declare (type fixnum run))
                   (when (> r 0)
                     (incf run (recieve r nextbit image)))
                   ;; Apply correction bits to remaining nonzeros in this block
                   (loop while (<= k se) do
                         (apply-correction (+ offset k))
                         (incf k))
                   (return-from decode-ac-refine (1- run)))))))
      ;; Normal completion (no EOB)
      0)))

;;; Main progressive chunk decoder - processes one restart interval
(defun decode-progressive-chunk (image scan)
  "Decode one restart interval of a progressive scan into coefficient buffers."
  (let* ((nextbit (make-nextbit 0 0))
         (ncomp (scan-ncomp scan))
         (ss (scan-ss scan))
         (se (scan-se scan))
         (ah (scan-ah scan))
         (al (scan-al scan))
         (dc-scan-p (and (zerop ss) (zerop se)))
         (first-scan-p (zerop ah))
         ;; Build component mapping: scan component index -> image component index
         (comp-indices (make-array ncomp :element-type 'fixnum))
         ;; Sampling factors for components in this scan
         (fr (make-array ncomp :initial-contents
                         (loop for i fixnum from 0 below ncomp
                               for cid fixnum = (first (aref (scan-cdesc scan) i))
                               for pos fixnum = (position cid (descriptor-cid image))
                               do (setf (aref comp-indices i) pos)
                               collecting (list (aref (descriptor-H image) pos)
                                                (aref (descriptor-V image) pos)))))
         (Hmax (loop for entry across fr maximize (the fixnum (first entry))))
         (Vmax (loop for entry across fr maximize (the fixnum (second entry))))
         (mcus-x (ceiling (descriptor-width image) (* Hmax 8)))
         (mcus-y (ceiling (descriptor-height image) (* Vmax 8)))
         ;; Huffman tables for each component in scan
         ;; huff-ac stores DC tables (tc=0), huff-dc stores AC tables (tc=1)
         ;; For DC scans use Td (high nibble) to select DC table from huff-ac
         ;; For AC scans use Ta (low nibble) to select AC table from huff-dc
         (dc-tables (when dc-scan-p
                      (make-array ncomp
                        :initial-contents
                        (loop for i fixnum from 0 below ncomp
                              for td fixnum = (ash (the fixnum (second (aref (scan-cdesc scan) i))) -4)
                              collecting (aref (descriptor-huff-ac image) td)))))
         (ac-tables (when (not dc-scan-p)
                      (make-array ncomp
                        :initial-contents
                        (loop for i fixnum from 0 below ncomp
                              for ta fixnum = (logand (second (aref (scan-cdesc scan) i)) 15)
                              collecting (aref (descriptor-huff-dc image) ta)))))
         (preds (make-array ncomp :initial-element 0 :element-type 'sint16))
         (eobrun 0))
    (declare #.*optimize*
             (type fixnum ncomp ss se ah al Hmax Vmax mcus-x mcus-y eobrun)
             (type (simple-array fixnum (*)) comp-indices)
             (type (simple-array sint16 (*)) preds))
    ;; scan-x is used as a linear MCU/block counter to persist across restart intervals
    (catch 'marker
      (if (= ncomp 1)
          ;; Non-interleaved scan: raster order over all blocks of one component
          (let* ((ci (aref comp-indices 0))
                 (coef-buf (aref (descriptor-coef-buffers image) ci))
                 (bpr (aref (descriptor-blocks-per-row image) ci))
                 (bpc (aref (descriptor-blocks-per-col image) ci))
                 (total-blocks (* bpr bpc))
                 (dc-tab (when dc-tables (aref dc-tables 0)))
                 (ac-tab (when ac-tables (aref ac-tables 0))))
            (declare (type fixnum ci bpr bpc total-blocks)
                     (type (simple-array fixnum (*)) coef-buf))
            (loop for block-idx fixnum from (scan-x scan) below total-blocks
                  for offset fixnum = (* block-idx 64) do
                  (cond
                    ;; DC first
                    ((and dc-scan-p first-scan-p)
                     (decode-dc-first coef-buf offset al dc-tab nextbit image preds 0))
                    ;; DC refine
                    (dc-scan-p
                     (decode-dc-refine coef-buf offset al nextbit image))
                    ;; AC first
                    (first-scan-p
                     (setf eobrun (decode-ac-first coef-buf offset ss se al ac-tab nextbit image eobrun)))
                    ;; AC refine
                    (t
                     (setf eobrun (decode-ac-refine coef-buf offset ss se al ac-tab nextbit image eobrun))))
                  (setf (scan-x scan) (1+ block-idx))))
          ;; Interleaved scan (DC only - AC scans must be non-interleaved per spec)
          (let ((total-mcus (* mcus-x mcus-y)))
            (declare (type fixnum total-mcus))
            (loop for mcu-idx fixnum from (scan-x scan) below total-mcus
                  for mcu-x fixnum = (mod mcu-idx mcus-x)
                  for mcu-y fixnum = (floor mcu-idx mcus-x) do
                  (loop for comp fixnum from 0 below ncomp
                        for ci fixnum = (aref comp-indices comp)
                        for coef-buf = (aref (descriptor-coef-buffers image) ci)
                        for bpr fixnum = (aref (descriptor-blocks-per-row image) ci)
                        for blocks-y fixnum = (second (aref fr comp))
                        for blocks-x fixnum = (first (aref fr comp))
                        for dc-tab = (aref dc-tables comp) do
                        (loop for by fixnum from 0 below blocks-y do
                              (loop for bx fixnum from 0 below blocks-x
                                    for block-row fixnum = (+ (* mcu-y blocks-y) by)
                                    for block-col fixnum = (+ (* mcu-x blocks-x) bx)
                                    for block-idx fixnum = (+ (* block-row bpr) block-col)
                                    for offset fixnum = (* block-idx 64) do
                                    (if first-scan-p
                                        (decode-dc-first coef-buf offset al dc-tab nextbit image preds comp)
                                        (decode-dc-refine coef-buf offset al nextbit image)))))
                  (setf (scan-x scan) (1+ mcu-idx))))))))

;;; Reads SOS header and decodes one progressive scan
(defun decode-progressive-scan (image)
  "Read SOS header and decode one progressive scan into coefficient buffers."
  (let ((scan (make-scan)))
    (read-jpeg-byte image) ; length
    (read-jpeg-byte image)
    (let ((ncomp (read-jpeg-byte image)))
      (setf (scan-ncomp scan) ncomp)
      (loop for j fixnum from 0 below ncomp do
            (setf (first (aref (scan-cdesc scan) j)) (read-jpeg-byte image))
            (setf (second (aref (scan-cdesc scan) j)) (read-jpeg-byte image))))
    (setf (scan-ss scan) (read-jpeg-byte image))
    (setf (scan-se scan) (read-jpeg-byte image))
    (let ((ahal (read-jpeg-byte image)))
      (setf (scan-ah scan) (ash ahal -4))
      (setf (scan-al scan) (logand ahal 15)))
    (if (= (descriptor-restart-interval image) 0)
        (decode-progressive-chunk image scan)
      (loop for term = (decode-progressive-chunk image scan)
            while (eq 'restart term)
            finally (return term)))))

;;; Convert accumulated coefficients to pixel data
(defun finalize-progressive (image)
  "Run IDCT on all accumulated coefficients and write to pixel buffer."
  (let* ((ncomp (descriptor-ncomp image))
         (nwidth (mul (descriptor-width image) ncomp))
         (dend (mul (descriptor-height image) nwidth))
         (zzbuf (make-array 64 :element-type 'sint16 :initial-element 0))
         (block-2d (2d-sint16-array
                    '(0  0  0  0  0  0  0  0)
                    '(0  0  0  0  0  0  0  0)
                    '(0  0  0  0  0  0  0  0)
                    '(0  0  0  0  0  0  0  0)
                    '(0  0  0  0  0  0  0  0)
                    '(0  0  0  0  0  0  0  0)
                    '(0  0  0  0  0  0  0  0)
                    '(0  0  0  0  0  0  0  0)))
         (frl (loop for i fixnum from 0 below ncomp
                    collecting (list (aref (descriptor-H image) i)
                                     (aref (descriptor-V image) i))))
         (Hmax (loop for entry in frl maximize (first entry)))
         (Vmax (loop for entry in frl maximize (second entry)))
         (freqs (convert-sampling frl Hmax Vmax))
         (scan (make-scan)))
    (declare (type fixnum ncomp nwidth dend Hmax Vmax)
             (type sint16-array zzbuf)
             (type sint16-2d-array block-2d))
    ;; Set up inverse sampling frequencies
    (loop for entry across freqs
          for i fixnum from 0 do
          (setf (aref (descriptor-iH image) i) (first entry))
          (setf (aref (descriptor-iV image) i) (second entry)))
    ;; Process blocks in MCU order (same order as decode-chunk/upsample expects)
    (let ((mcus-x (ceiling (descriptor-width image) (mul Hmax 8)))
          (mcus-y (ceiling (descriptor-height image) (mul Vmax 8))))
      (declare (type fixnum mcus-x mcus-y))
      (setf (scan-x scan) 0 (scan-y scan) 0)
      (loop for mcu-y fixnum from 0 below mcus-y do
            (loop for mcu-x fixnum from 0 below mcus-x do
                  ;; Progressive mode: coefficient buffers are indexed by frame component index 0..ncomp-1,
                  ;; which directly matches (descriptor-cid image) order (comp-index: 0->Y, 1->Cb, 2->Cr).
                  (loop for comp-index fixnum from 0 below ncomp
                        for coef-buf = (aref (descriptor-coef-buffers image) comp-index)
                        for bpr fixnum = (aref (descriptor-blocks-per-row image) comp-index)
                        for q-tab = (aref (descriptor-qtables image)
                                          (aref (descriptor-qdest image) comp-index))
                        for h-samp fixnum = (aref (descriptor-H image) comp-index)
                        for v-samp fixnum = (aref (descriptor-V image) comp-index)
                        for H fixnum = (first (aref freqs comp-index))
                        for V fixnum = (second (aref freqs comp-index))
                        for nw fixnum = (mul nwidth V)
                        for nx fixnum = (mul ncomp H) do
                        (loop for by fixnum from 0 below v-samp
                              for y-pos fixnum from (mul (ash by 3) V) by (ash V 3) do
                              (loop for bx fixnum from 0 below h-samp
                                    for x-pos fixnum from (mul (ash bx 3) H) by (ash H 3)
                                    for block-row fixnum = (plus (mul mcu-y v-samp) by)
                                    for block-col fixnum = (plus (mul mcu-x h-samp) bx)
                                    for block-idx fixnum = (plus (mul block-row bpr) block-col)
                                    for offset fixnum = (mul block-idx 64) do
                                    ;; Copy coefficients to zigzag buffer
                                    (loop for k fixnum from 0 below 64 do
                                          (setf (aref zzbuf k)
                                                (the sint16 (aref coef-buf (plus offset k)))))
                                    ;; Inverse zigzag to 8x8
                                    (izigzag zzbuf block-2d)
                                    ;; Bounds check before IDCT and upsample
                                    (when (and (< (plus x-pos (scan-x scan)) (descriptor-width image))
                                               (< (plus y-pos (scan-y scan)) (descriptor-height image)))
                                      ;; IDCT + dequantization
                                      (inverse-llm-dct block-2d q-tab (descriptor-ws image))
                                      ;; Note: upsample adds (scan-x scan) + x-pos and (scan-y scan) + y-pos.
                                      ;; scan-x/scan-y track the MCU origin, while x-pos/y-pos are intra-MCU offsets.
                                      ;; If scan-x/scan-y were left at 0, x-pos/y-pos would have to be absolute positions;
                                      ;; ensure coordinates are not double-counted in any future refactoring.
                                      (upsample image scan block-2d x-pos y-pos
                                                H V comp-index nwidth nw nx dend)))))
                  (incf (scan-x scan) (ash Hmax 3))
                  (when (<= (descriptor-width image) (scan-x scan))
                    (incf (scan-y scan) (ash Vmax 3))
                    (setf (scan-x scan) 0)))))))

(defun setup-planar-buffers-if-supported (image colorspace-conversion)
  (declare #.*optimize*
           (type descriptor image))
  (setf (descriptor-colorspace-conversion-p image) (if colorspace-conversion t nil))
  (if (and colorspace-conversion
           *upsample-impl*
           (not (eql *use-simd* :off))
           (= (descriptor-ncomp image) 3)
           (let ((h-arr (descriptor-H image))
                 (v-arr (descriptor-V image)))
             (or (and (= (the fixnum (aref h-arr 0)) 2) (= (the fixnum (aref v-arr 0)) 2)
                      (= (the fixnum (aref h-arr 1)) 1) (= (the fixnum (aref v-arr 1)) 1)
                      (= (the fixnum (aref h-arr 2)) 1) (= (the fixnum (aref v-arr 2)) 1))
                 (and (= (the fixnum (aref h-arr 0)) 1) (= (the fixnum (aref v-arr 0)) 1)
                      (= (the fixnum (aref h-arr 1)) 1) (= (the fixnum (aref v-arr 1)) 1)
                      (= (the fixnum (aref h-arr 2)) 1) (= (the fixnum (aref v-arr 2)) 1)))))
      (let* ((img-w (descriptor-width image))
             (img-h (descriptor-height image))
             (h-arr (descriptor-H image))
             ;; For 4:2:0: chroma planes are half-width, half-height.
             ;; For 4:4:4: chroma planes are full-width, full-height.
             (is-420 (= (the fixnum (aref h-arr 0)) 2))
             (chroma-w (if is-420 (the fixnum (ash (the fixnum (1+ img-w)) -1)) img-w))
             (chroma-h (if is-420 (the fixnum (ash (the fixnum (1+ img-h)) -1)) img-h))
             (y-req-size (the fixnum (+ (the fixnum (* img-w img-h)) 64)))
             (chroma-req-size (the fixnum (+ (the fixnum (* chroma-w chroma-h)) 64))))
        (declare (type fixnum img-w img-h chroma-w chroma-h y-req-size chroma-req-size))
        (setf (descriptor-planar-p image) t)
        ;; Note on lazy allocation: planes are allocated to accommodate the image dimensions,
        ;; but are not shrunk if the descriptor is subsequently reused for a smaller image.
        ;; For example, a descriptor reused from a 4K decode to a thumbnail decode retains its
        ;; allocated plane buffers (6 MB) to avoid repeated garbage-collection churn.
        (unless (and (descriptor-y-plane image)
                     (>= (length (the uint8-array (descriptor-y-plane image))) y-req-size))
          (setf (descriptor-y-plane image)
                (make-array y-req-size :element-type 'uint8)))
        (unless (and (descriptor-cb-plane image)
                     (>= (length (the uint8-array (descriptor-cb-plane image))) chroma-req-size))
          (setf (descriptor-cb-plane image)
                (make-array chroma-req-size :element-type 'uint8)))
        (unless (and (descriptor-cr-plane image)
                     (>= (length (the uint8-array (descriptor-cr-plane image))) chroma-req-size))
          (setf (descriptor-cr-plane image)
                (make-array chroma-req-size :element-type 'uint8))))
      (setf (descriptor-planar-p image) nil)))

;;; Progressive frame decoding subroutine
(defun decode-progressive-frame (image buffer &optional (colorspace-conversion t))
  "Decode a progressive JPEG frame (SOF2)."
  (decode-frame-beginning image buffer)
  (loop for i fixnum from 0 below (descriptor-ncomp image)
        with hv fixnum do
        (setf (aref (descriptor-cid image) i) (read-jpeg-byte image))
        (setf hv (read-jpeg-byte image))
        (setf (aref (descriptor-H image) i) (ash hv -4))
        (setf (aref (descriptor-V image) i) (logand hv 7))
        (setf (aref (descriptor-qdest image) i) (read-jpeg-byte image)))
  (let* ((frl (loop for i fixnum from 0 below (descriptor-ncomp image)
                    collecting (list (aref (descriptor-H image) i)
                                     (aref (descriptor-V image) i))))
         (Hmax (loop for entry in frl maximize (first entry)))
         (Vmax (loop for entry in frl maximize (second entry))))
    ;; Allocate coefficient buffers
    (setf (descriptor-progressive-p image) t)
    (allocate-coef-buffers image Hmax Vmax)
    (setup-planar-buffers-if-supported image colorspace-conversion)
    ;; Process scans until EOI
    (loop with term fixnum = 0 do
          (let ((marker (interpret-markers image term)))
            (cond ((= marker +M_SOS+)
                   (setf term (or (decode-progressive-scan image) 0)))
                  ((= marker +M_EOI+)
                   (return))
                  (t (error 'unsupported-jpeg-frame-marker)))))
    ;; All scans done - finalize: IDCT and write pixels
    (finalize-progressive image)))

;;; Scan decoding subroutine
(defun decode-scan (image i)
  (let ((scan (aref (descriptor-scans image) i))
	(zzbuf (2d-sint16-array
			    '(0  0  0  0  0  0  0  0)
			    '(0  0  0  0  0  0  0  0)
			    '(0  0  0  0  0  0  0  0)
			    '(0  0  0  0  0  0  0  0)
			    '(0  0  0  0  0  0  0  0)
			    '(0  0  0  0  0  0  0  0)
			    '(0  0  0  0  0  0  0  0)
			    '(0  0  0  0  0  0  0  0))))
    (read-jpeg-byte image) ; length
    (read-jpeg-byte image)
    (loop with ncomp fixnum = (setf (scan-ncomp scan) (read-jpeg-byte image))
          for j fixnum from 0 below ncomp do
          (setf (first (aref (scan-cdesc scan) j)) (read-jpeg-byte image)) ; component ID
          (setf (second (aref (scan-cdesc scan) j)) (read-jpeg-byte image))) ; Td and Ta nibbles
    (setf (scan-ss scan) (read-jpeg-byte image))          ; spectral selection start
    (setf (scan-se scan) (read-jpeg-byte image))          ; spectral selection end
    (let ((ahal (read-jpeg-byte image)))                   ; successive approximation
      (setf (scan-ah scan) (ash ahal -4))
      (setf (scan-al scan) (logand ahal 15)))
    (if (= (descriptor-restart-interval image) 0)
        (decode-chunk image scan zzbuf) ; reading the whole scan at once
      (loop for term = (decode-chunk image scan zzbuf)
            while (eq 'restart term)
            finally (return term))))) ; or in pieces

;;; Function that bounds value in 0..255 range
(defun limit (n)
  (cond ((> n 254) 255)
        ((< n 1) 0)
        (t n)))

;;; SIMD Dispatch Hook (Controls declared at top of file)

(defun inverse-colorspace-convert-dispatch (descriptor)
  (declare (optimize (speed 3) (safety 0)))
  (ecase *use-simd*
    (:off (inverse-colorspace-convert descriptor))
    ((:auto :on) (funcall (the function (or *inverse-colorspace-convert-impl* #'inverse-colorspace-convert)) descriptor))))

;;; Inverse colorspace conversion
(defun inverse-colorspace-convert (image)
  (let* ((buffer (descriptor-buffer image))
         (nw (mul (descriptor-width image) 3)))
    (declare #.*optimize*
             (type (simple-array uint8 (*)) buffer)
             (type fixnum nw))
    (loop for y fixnum from 0 below (descriptor-height image)
          for yp fixnum from 0 by nw do
          (loop for x fixnum from 0 below (descriptor-width image)
                for py fixnum from yp by 3
                for pu fixnum = (1+ py)
                for pv fixnum = (plus py 2)
                for yy fixnum = (aref buffer py)
                for cb fixnum = (aref buffer pu)
                for cr fixnum = (aref buffer pv) do
                (setf (aref buffer py) ; BLUE
                      (the uint8 (limit (plus yy (aref +cb-b-tab+ cb)))))
                (setf (aref buffer pu) ; GREEN
                      (the uint8 (limit (plus yy (ash (plus
						       (aref +cb-g-tab+ cb)
						       (aref +cr-g-tab+ cr))
						      (- shift))))))
                (setf (aref buffer pv) ; RED
                      (the uint8 (limit (plus yy (aref +cr-r-tab+ cr)))))))))

(setf *inverse-colorspace-convert-impl* #'inverse-colorspace-convert)

(defun ycck-cmyk-convert (image)
  (let* ((buffer (descriptor-buffer image))
         (nw (mul (descriptor-width image) 4)))
    (declare #.*optimize*
	     (type (simple-array uint8 (*)) buffer)
             (type fixnum nw))
    (loop for y fixnum from 0 below (descriptor-height image)
       for yp fixnum from 0 by nw do
	 (loop for x fixnum from 0 below (descriptor-width image)
	    for py fixnum from yp by 4
	    for pu fixnum = (1+ py)
	    for pv fixnum = (plus py 2)
	    for yy fixnum = (aref buffer py)
	    for cb fixnum = (aref buffer pu)
	    for cr fixnum = (aref buffer pv) do
	      (setf (aref buffer pv)	; BLUE
		    (the uint8 (limit (minus +max-sample+ (plus yy (aref +cb-b-tab+ cb))))))
	      (setf (aref buffer pu)	; GREEN
		    (the uint8 (limit (minus +max-sample+
					     (plus yy (ash (plus
							    (aref +cb-g-tab+ cb)
							    (aref +cr-g-tab+ cr))
							   (- shift)))))))
	      (setf (aref buffer py)	; RED
		    (the uint8 (limit (minus +max-sample+ (plus yy (aref +cr-r-tab+ cr))))))))))

(defun convert-cmyk-to-rgb (buffer h w &key rgb-buffer)
  (unless rgb-buffer
    (setf rgb-buffer (make-array (* h w 3) :element-type 'uint8)))
  (loop for y fixnum from 0 below h
     for yrgb fixnum from 0 by (* w 3)
     for yp fixnum from 0 by (* w 4) do
       (loop for x fixnum from 0 below w
	  for py fixnum from yp by 4
	  for pu fixnum = (1+ py)
	  for pv fixnum = (plus py 2)
	  for pb fixnum from yrgb by 3
	  for pg fixnum = (1+ pb)
	  for pr fixnum = (plus pb 2)
	  for cyan fixnum = (aref buffer py)
	  for magenta fixnum = (aref buffer pu)
	  for yellow fixnum = (aref buffer pv)
	  for black fixnum = (aref buffer (1+ pv)) do
	    (setf (aref rgb-buffer pr) (round (mul cyan black) 255)
		  (aref rgb-buffer pg) (round (mul magenta black) 255)
		  (aref rgb-buffer pb) (round (mul yellow black) 255))))
  rgb-buffer)

(defun allocate-buffer (height width ncomp)
  (make-array (+ (* height width ncomp) 64)
	      :element-type 'uint8
	      :initial-element 0))

(defun decode-frame-beginning (image buffer)
  (read-jpeg-byte image) ; length
  (read-jpeg-byte image)
  (read-jpeg-byte image) ; sample precision
  (let ((height (read-word image))
        (width (read-word image))
        (ncomp (read-jpeg-byte image)))
    (setf (descriptor-height image) height
          (descriptor-width image) width
          (descriptor-ncomp image) ncomp)
    (let ((req-buf-size (the fixnum (+ (the fixnum (* height (the fixnum (* width ncomp)))) 64))))
      (declare (type fixnum req-buf-size))
      (if (arrayp buffer)
          (if (< (length (the (simple-array uint8 (*)) buffer)) (the fixnum (* height (the fixnum (* width ncomp)))))
              (error 'invalid-buffer-supplied :buffer buffer)
              (setf (descriptor-buffer image) buffer))
          (unless (and (descriptor-buffer image)
                       (>= (length (the (simple-array uint8 (*)) (descriptor-buffer image))) req-buf-size))
            (setf (descriptor-buffer image)
                  (allocate-buffer height width ncomp)))))))

;;; Frame decoding subroutine
(defun decode-frame (image buffer &optional (colorspace-conversion t))
  (decode-frame-beginning image buffer)
  (loop for i fixnum from 0 below (descriptor-ncomp image)
     with hv fixnum do
       (setf (aref (descriptor-cid image) i) (read-jpeg-byte image)) ; Cj
       (setf hv (read-jpeg-byte image))				    ; HV
       (setf (aref (descriptor-H image) i) (ash hv -4))
       (setf (aref (descriptor-V image) i) (logand hv 7))
       (setf (aref (descriptor-qdest image) i) (read-jpeg-byte image)))
	(let* ((frl (loop for i fixnum from 0 below (descriptor-ncomp image)
		       collecting (list (aref (descriptor-H image) i)
					(aref (descriptor-V image) i))))
	       (Hmax (loop for entry in frl maximize (first entry)))
	       (Vmax (loop for entry in frl maximize (second entry)))
	       (freqs (convert-sampling frl Hmax Vmax)))
	  (loop for entry across freqs
	     for i fixnum from 0 do
	       (setf (aref (descriptor-iH image) i) (first entry))
	       (setf (aref (descriptor-iV image) i) (second entry)))
          (setup-planar-buffers-if-supported image colorspace-conversion)
	  (loop with term fixnum = 0
	     for j fixnum from 0
	     until (= term +M_EOI+) do
	       (when (/= (interpret-markers image term) +M_SOS+)
		 (error 'unsupported-jpeg-frame-marker))
	       (setf term (decode-scan image j)))))

(defun decode-stream (stream &key buffer (colorspace-conversion t) descriptor cached-source-p (decode-frame t))
  "Return image array, height, width, number of components and APP14 Adobe transform.
Supports both baseline and progressive DCT-based JPEGs."
  (when (and (null stream) (not cached-source-p))
    (error 'invalid-buffer-supplied))
  (when descriptor
    (setf (descriptor-restart-interval descriptor) 0
          (descriptor-adobe-app14-transform descriptor) nil
          (descriptor-progressive-p descriptor) nil
          (descriptor-bitreader descriptor) nil
          (descriptor-ncomp descriptor) 0)
    (loop for scan across (descriptor-scans descriptor) do ;required if we reuse descriptors
	 (setf (scan-x scan) 0
	       (scan-y scan) 0)))
  (let* ((image (or descriptor (make-descriptor)))
	 (pos 0))
    (cond (cached-source-p
           (when (or (not (typep (descriptor-source-cache image) 'array))
                     (and (typep stream 'file-stream) (< (length (descriptor-source-cache image)) (file-length stream))))
             (setf (descriptor-source-cache image) (make-array (file-length stream) :element-type 'uint8)))
           (when stream ;; NULL stream means the cache in descriptor is already read
             (read-sequence (descriptor-source-cache image) stream))
           (let ((cache (descriptor-source-cache image)))
             (setf (descriptor-byte-reader image)
                   #'(lambda ()
                       (declare #.*optimize*
                                (type uint8-array cache)
                                (type fixnum pos))
                       (prog1 (aref cache pos) (incf pos))))))
          (t
           (setf (descriptor-byte-reader image)
                 #'(lambda ()
                     (read-byte stream)))))
     (unless (= (read-marker image) +M_SOI+)
      (error 'unrecognized-file-format))
     (let ((marker (interpret-markers image 0)))
       (if decode-frame
           (cond ((= +M_SOF0+ marker)
                  (decode-frame image buffer colorspace-conversion)
                  (when colorspace-conversion
                    (cond ((and (= (descriptor-ncomp image) 3) (eql (descriptor-adobe-app14-transform image) :ycbcr-rgb))
                           (inverse-colorspace-convert-dispatch image))
                          ((eql (descriptor-adobe-app14-transform image) :ycck-cmyk)
                            (ycck-cmyk-convert image))
                          ((= (descriptor-ncomp image) 3)
                           (inverse-colorspace-convert-dispatch image)))))
                  ((= +M_SOF2+ marker)
                  (decode-progressive-frame image buffer colorspace-conversion)
                  (when colorspace-conversion
                    (cond ((and (= (descriptor-ncomp image) 3) (eql (descriptor-adobe-app14-transform image) :ycbcr-rgb))
                           (inverse-colorspace-convert-dispatch image))
                          ((eql (descriptor-adobe-app14-transform image) :ycck-cmyk)
                            (ycck-cmyk-convert image))
                          ((= (descriptor-ncomp image) 3)
                           (inverse-colorspace-convert-dispatch image)))))
                  (t (error 'unsupported-jpeg-format :code marker)))
           ;; decode-frame-beginning supports baselnie DCT frames
           ;; and progressive DCT frames, so allow those
           (cond ((or (= +M_SOF0+ marker)
                      (= +M_SOF2+ marker))
                  (decode-frame-beginning image nil))
                 (t (error 'unsupported-jpeg-format :code marker)))))
     (values (when decode-frame
               (descriptor-buffer image))
             (descriptor-height image)
             (descriptor-width image)
             (descriptor-ncomp image)
             (descriptor-adobe-app14-transform image))))

;;; Top level decoder function
(defun decode-image (filename &key buffer (colorspace-conversion t) descriptor cached-source-p (decode-frame t))
  (with-open-file (in filename :direction :input :element-type 'uint8)
    (decode-stream in
                   :buffer buffer
                   :colorspace-conversion colorspace-conversion
                   :descriptor descriptor
                   :cached-source-p cached-source-p
                   :decode-frame decode-frame)))

(defun decode-stream-height-width-ncomp (stream &key descriptor cached-source-p)
  "Return the height and width of the JPEG data read from STREAM. Does less work than
DECODE-STREAM and also supports progressive DCT-based JPEGs."
  (multiple-value-bind (nil-buffer height width ncomp adobe-app14-transform)
      (apply #'decode-stream stream
             :decode-frame nil
             (append
              (when descriptor
                `(:descriptor ,descriptor))
              (when cached-source-p
                `(:cached-source-p ,cached-source-p))))
    (declare (ignore nil-buffer))
    (values height width ncomp adobe-app14-transform)))

(defun jpeg-file-dimensions (filename)
  "Return image height, width and number of components, plus the type of Adobe colorpsace transform"
  (with-open-file (in filename :direction :input :element-type 'uint8)
    (decode-stream-height-width-ncomp in)))
