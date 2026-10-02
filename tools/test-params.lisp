(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(dolist (f '("/tmp/photo1_512x512.jpg" "/tmp/photo3_1024x1024.jpg"))
  (with-open-file (stream f :direction :input :element-type 'uint8)
    (let ((image (make-descriptor)))
      (setf (descriptor-byte-reader image) (lambda () (read-byte stream nil 0)))
      (read-marker image)
      (interpret-markers image 0)
      (decode-frame-beginning image nil)
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
             (Vmax (loop for entry in frl maximize (second entry)))
             (freqs (convert-sampling frl Hmax Vmax)))
        (format t "~%File: ~A~%" f)
        (format t "  Dims: ~Dx~D, ncomp: ~D, Hmax=~D, Vmax=~D~%"
                (descriptor-width image) (descriptor-height image) (descriptor-ncomp image) Hmax Vmax)
        (dotimes (c (descriptor-ncomp image))
          (format t "  Comp ~D: cid=~D, H=~D, V=~D, freq=~S~%"
                  c
                  (aref (descriptor-cid image) c)
                  (aref (descriptor-H image) c)
                  (aref (descriptor-V image) c)
                  (aref freqs c)))))))

(sb-ext:exit :code 0)
