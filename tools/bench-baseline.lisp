;;; bench-baseline.lisp - CL-JPEG Baseline Benchmark & Hot Path Profiling
(require :asdf)
(load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname)))
(push (truename "/home/arthur/repos/cl/cl-jpeg-avx512") asdf:*central-registry*)
(asdf:load-system "cl-jpeg")

(in-package :jpeg)

(defun make-test-image (w h)
  "Create synthetic RGB image with color gradients."
  (let ((buf (make-array (* w h 3) :element-type 'uint8)))
    (dotimes (y h)
      (dotimes (x w)
        (let ((idx (* (+ (* y w) x) 3)))
          (setf (aref buf idx)       (logand (+ x y) 255)         ; B
                (aref buf (+ idx 1)) (logand (* x 2) 255)         ; G
                (aref buf (+ idx 2)) (logand (* y 2) 255)))))     ; R
    buf))

(defun bench-colorspace-convert (w h &key (iterations 20))
  (let* ((rgb (make-test-image w h))
         (YUV (make-array 3 :initial-contents
                          (loop repeat 3 collect
                                (make-array h :initial-contents
                                            (loop repeat h collect
                                                  (make-array w :element-type 'sint16 :initial-element 0))))))
         (mpixels (/ (* w h) 1000000.0)))
    ;; Warmup
    (colorspace-convert rgb YUV 0 0 h w h w)
    (let ((start (get-internal-real-time)))
      (dotimes (i iterations)
        (colorspace-convert rgb YUV 0 0 h w h w))
      (let* ((elapsed (/ (- (get-internal-real-time) start)
                         (float internal-time-units-per-second)))
             (per-iter (/ elapsed iterations))
             (mp-per-sec (/ mpixels per-iter)))
        (format t "  RGB->YCbCr (Encoder) (~Dx~D, ~D trials): ~8,3F ms/iter (~6,1F Mpixels/sec)~%"
                w h iterations (* per-iter 1000.0) mp-per-sec)
        (values per-iter mp-per-sec)))))

(defun bench-inverse-colorspace (w h &key (iterations 20))
  (let* ((desc (make-descriptor))
         (npixels (* w h))
         (mpixels (/ npixels 1000000.0))
         (out-buf (make-test-image w h)))
    (setf (descriptor-width desc) w
          (descriptor-height desc) h
          (descriptor-ncomp desc) 3
          (descriptor-buffer desc) out-buf)
    ;; Warmup
    (inverse-colorspace-convert desc)
    (let ((start (get-internal-real-time)))
      (dotimes (i iterations)
        (inverse-colorspace-convert desc))
      (let* ((elapsed (/ (- (get-internal-real-time) start)
                         (float internal-time-units-per-second)))
             (per-iter (/ elapsed iterations))
             (mp-per-sec (/ mpixels per-iter)))
        (format t "  YCbCr->RGB (Decoder) (~Dx~D, ~D trials): ~8,3F ms/iter (~6,1F Mpixels/sec)~%"
                w h iterations (* per-iter 1000.0) mp-per-sec)
        (values per-iter mp-per-sec)))))

(defun bench-dct (nblocks &key (iterations 50))
  (let ((block (make-array 8 :initial-contents
                           (loop repeat 8 collect
                                 (make-array 8 :element-type 'sint16 :initial-element 64)))))
    (let ((start (get-internal-real-time)))
      (dotimes (it iterations)
        (dotimes (b nblocks)
          (llm-dct block)))
      (let* ((elapsed (/ (- (get-internal-real-time) start)
                         (float internal-time-units-per-second)))
             (total-blocks (* nblocks iterations))
             (blocks-per-sec (/ total-blocks elapsed)))
        (format t "  Forward LLM-DCT (~D blocks, ~D trials): ~8,3F us/block (~8,0F blocks/sec)~%"
                nblocks iterations (* (/ elapsed total-blocks) 1000000.0) blocks-per-sec)))))

(defun bench-idct (nblocks &key (iterations 50))
  (let ((block (make-array 8 :initial-contents
                           (loop repeat 8 collect
                                 (make-array 8 :element-type 'sint16 :initial-element 64))))
        (ws (make-array 8 :initial-contents
                        (loop repeat 8 collect
                              (make-array 8 :element-type 'sint16 :initial-element 0)))))
    (let ((start (get-internal-real-time)))
      (dotimes (it iterations)
        (dotimes (b nblocks)
          (inverse-llm-dct block +q-luminance+ ws)))
      (let* ((elapsed (/ (- (get-internal-real-time) start)
                         (float internal-time-units-per-second)))
             (total-blocks (* nblocks iterations))
             (blocks-per-sec (/ total-blocks elapsed)))
        (format t "  Inverse LLM-DCT (~D blocks, ~D trials): ~8,3F us/block (~8,0F blocks/sec)~%"
                nblocks iterations (* (/ elapsed total-blocks) 1000000.0) blocks-per-sec)))))

(defun bench-end-to-end (w h &key (iterations 10))
  (let* ((rgb (make-test-image w h))
         (out-path (format nil "/tmp/cl-jpeg-bench-~Dx~D.jpg" w h))
         (mpixels (/ (* w h) 1000000.0)))
    ;; Encode
    (encode-image (pathname out-path) rgb 3 h w)
    (let ((start (get-internal-real-time)))
      (dotimes (i iterations)
        (encode-image (pathname out-path) rgb 3 h w))
      (let* ((elapsed (/ (- (get-internal-real-time) start)
                         (float internal-time-units-per-second)))
             (per-iter (/ elapsed iterations))
             (mp-per-sec (/ mpixels per-iter)))
        (format t "  Full ENCODE (~Dx~D): ~8,3F ms (~6,1F Mpixels/sec)~%"
                w h (* per-iter 1000.0) mp-per-sec)))
    ;; Decode
    (decode-image (pathname out-path))
    (let ((start (get-internal-real-time)))
      (dotimes (i iterations)
        (decode-image (pathname out-path)))
      (let* ((elapsed (/ (- (get-internal-real-time) start)
                         (float internal-time-units-per-second)))
             (per-iter (/ elapsed iterations))
             (mp-per-sec (/ mpixels per-iter)))
        (format t "  Full DECODE (~Dx~D): ~8,3F ms (~6,1F Mpixels/sec)~%"
                w h (* per-iter 1000.0) mp-per-sec)))))

(defun run-all-benchmarks ()
  (format t "~%================================================================~%")
  (format t " CL-JPEG Baseline Benchmark (Before AVX-512 SIMD)~%")
  (format t " Platform: SBCL ~A on AMD Zen 5~%" (lisp-implementation-version))
  (format t "================================================================~%~%")

  (format t "--- 1. Color Conversion Micro-benchmarks ---~%")
  (bench-colorspace-convert 512 512 :iterations 20)
  (bench-colorspace-convert 1024 1024 :iterations 10)
  (bench-colorspace-convert 2048 2048 :iterations 5)
  (terpri)
  (bench-inverse-colorspace 512 512 :iterations 20)
  (bench-inverse-colorspace 1024 1024 :iterations 10)
  (bench-inverse-colorspace 2048 2048 :iterations 5)

  (format t "~%--- 2. Discrete Cosine Transform (DCT) Micro-benchmarks ---~%")
  (bench-dct 1024 :iterations 20)
  (bench-dct 4096 :iterations 10)
  (terpri)
  (bench-idct 1024 :iterations 20)
  (bench-idct 4096 :iterations 10)

  (format t "~%--- 3. End-to-End Encode / Decode Benchmark ---~%")
  (bench-end-to-end 512 512 :iterations 10)
  (bench-end-to-end 1024 1024 :iterations 5)
  (bench-end-to-end 2048 2048 :iterations 3)
  (format t "~%================================================================~%"))

(run-all-benchmarks)
