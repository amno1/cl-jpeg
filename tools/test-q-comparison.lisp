(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :cl-user)

(let* ((raw1 "/home/arthur/repos/cl/cl-jpeg-avx512/tools/test-images/photo1_512x512.raw")
       (raw3 "/home/arthur/repos/cl/cl-jpeg-avx512/tools/test-images/photo3_1024x1024.raw")
       (buf1 (make-array (* 512 512 3) :element-type '(unsigned-byte 8)))
       (buf3 (make-array (* 1024 1024 3) :element-type '(unsigned-byte 8))))
  (with-open-file (s raw1 :element-type '(unsigned-byte 8))
    (read-sequence buf1 s))
  (with-open-file (s raw3 :element-type '(unsigned-byte 8))
    (read-sequence buf3 s))

  (jpeg:encode-image "/tmp/photo1_q64.jpg" buf1 3 512 512 :q-factor 64 :sampling '((2 2) (1 1) (1 1)))
  (jpeg:encode-image "/tmp/photo1_q25.jpg" buf1 3 512 512 :q-factor 25 :sampling '((2 2) (1 1) (1 1)))
  (jpeg:encode-image "/tmp/photo3_q64.jpg" buf3 3 1024 1024 :q-factor 64 :sampling '((2 2) (1 1) (1 1)))
  (jpeg:encode-image "/tmp/photo3_q25.jpg" buf3 3 1024 1024 :q-factor 25 :sampling '((2 2) (1 1) (1 1)))

  (format t "photo1 512x512: q=64 -> ~:D bytes, q=25 -> ~:D bytes (~,2Fx larger)~%"
          (with-open-file (s "/tmp/photo1_q64.jpg") (file-length s))
          (with-open-file (s "/tmp/photo1_q25.jpg") (file-length s))
          (/ (with-open-file (s "/tmp/photo1_q25.jpg") (file-length s))
             (float (with-open-file (s "/tmp/photo1_q64.jpg") (file-length s)))))

  (format t "photo3 1024x1024: q=64 -> ~:D bytes, q=25 -> ~:D bytes (~,2Fx larger)~%"
          (with-open-file (s "/tmp/photo3_q64.jpg") (file-length s))
          (with-open-file (s "/tmp/photo3_q25.jpg") (file-length s))
          (/ (with-open-file (s "/tmp/photo3_q25.jpg") (file-length s))
             (float (with-open-file (s "/tmp/photo3_q64.jpg") (file-length s))))))
(sb-ext:exit :code 0)
