(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun compute-psnr-test (buf1 buf2 size)
  (let ((mse 0.0d0))
    (dotimes (i size)
      (let ((diff (- (aref buf1 i) (aref buf2 i))))
        (incf mse (* diff diff))))
    (setf mse (/ mse size))
    (if (zerop mse) 99.0d0 (* 10.0d0 (log (/ (* 255.0d0 255.0d0) mse) 10.0d0)))))

(defun test-edge-clamp ()
  (let* ((w 227) (h 149)
         (src-ppm "/home/arthur/repos/libjpeg-turbo/testimages/testorig.ppm")
         (rgb-source (make-array (* w h 3) :element-type 'uint8))
         (ref-jpg "/tmp/ref_testorig_q80.jpg")
         (ref-ppm "/tmp/ref_testorig_q80.ppm")
         (cl-jpg "/tmp/cl_testorig_q25.jpg")
         (cl-ppm "/tmp/cl_testorig_q25.ppm"))
    (with-open-file (s src-ppm :element-type 'uint8)
      (read-line s) (read-line s) (read-line s)
      (read-sequence rgb-source s))
    (let ((bgr (make-array (* w h 3) :element-type 'uint8)))
      (dotimes (i (* w h))
        (setf (aref bgr (+ (* i 3) 0)) (aref rgb-source (+ (* i 3) 2))
              (aref bgr (+ (* i 3) 1)) (aref rgb-source (+ (* i 3) 1))
              (aref bgr (+ (* i 3) 2)) (aref rgb-source (+ (* i 3) 0))))
      
      ;; 1. Current CL-JPEG
      (encode-image cl-jpg bgr 3 h w :q-factor 25 :sampling '((2 2) (1 1) (1 1)) :chroma-subsampling :box)
      (uiop:run-program (format nil "djpeg -ppm ~A > ~A" cl-jpg cl-ppm) :force-shell t)
      
      ;; 2. Reference cjpeg
      (uiop:run-program (format nil "cjpeg -quality 80 ~A > ~A" src-ppm ref-jpg) :force-shell t)
      (uiop:run-program (format nil "djpeg -ppm ~A > ~A" ref-jpg ref-ppm) :force-shell t)

      (let* ((cl-sz (with-open-file (f cl-jpg) (file-length f)))
             (ref-sz (with-open-file (f ref-jpg) (file-length f)))
             (cl-dec (make-array (* w h 3) :element-type 'uint8))
             (ref-dec (make-array (* w h 3) :element-type 'uint8)))
        (with-open-file (s cl-ppm :element-type 'uint8)
          (read-line s) (read-line s) (read-line s)
          (read-sequence cl-dec s))
        (with-open-file (s ref-ppm :element-type 'uint8)
          (read-line s) (read-line s) (read-line s)
          (read-sequence ref-dec s))
        (let ((cl-psnr (compute-psnr-test rgb-source cl-dec (* w h 3)))
              (ref-psnr (compute-psnr-test rgb-source ref-dec (* w h 3))))
          (format t "Current: CL-JPEG ~D B (~,2F dB) vs cjpeg ~D B (~,2F dB), delta sz: ~+5,1F%, psnr: ~+5,2F dB~%"
                  cl-sz cl-psnr ref-sz ref-psnr
                  (* 100.0 (/ (- cl-sz ref-sz) (float ref-sz)))
                  (- cl-psnr ref-psnr)))))))

(test-edge-clamp)
(sb-ext:exit :code 0)
