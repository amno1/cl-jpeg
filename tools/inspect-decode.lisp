(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(format t "Checking decode-stream calling structure...~%")
(let ((path "/tmp/photo1_512x512.jpg"))
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((image (make-descriptor)))
      (decode-stream s :descriptor image)
      (format t "Image dims: ~Dx~D, ncomp: ~D~%"
              (descriptor-width image)
              (descriptor-height image)
              (descriptor-ncomp image))
      (format t "H: ~S, V: ~S, iH: ~S, iV: ~S~%"
              (descriptor-H image)
              (descriptor-V image)
              (descriptor-iH image)
              (descriptor-iV image))
      (format t "Adobe transform: ~S~%"
              (descriptor-adobe-app14-transform image)))))

(sb-ext:exit :code 0)
