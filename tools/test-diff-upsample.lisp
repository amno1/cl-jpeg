(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun test-upsamples ()
  (let* ((path "/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg")
         (buf-current (decode-image path)))
    
    ;; Redefine decode-chunk with original upsample
    (defun upsample-orig (image scan block x y H V offset nwidth nw nx dend)
      (let* ((buffer (descriptor-buffer image))
             (ncomp (descriptor-ncomp image))
             (xbase (plus (scan-x scan) x))
             (ybase (plus (scan-y scan) y))
             (nxbase (mul xbase ncomp))
             (nybase (mul ybase nwidth)))
        (declare #.*optimize*
                 (type sint16-2d-array block)
                 (type uint8-array buffer)
                 (type fixnum x y H V ncomp xbase ybase nwidth nx dend nxbase nybase offset))
        (loop for row of-type sint16-array across block
              for y fixnum from ybase below (descriptor-height image) by V
              for ypos fixnum from nybase by nw do
              (loop for val fixnum across row
                    for x fixnum from xbase below (descriptor-width image) by H
                    for pos fixnum from (the fixnum (+ ypos offset nxbase)) by nx do
                    (if (= 1 H V)
                        (setf (aref buffer pos) val)
                      (loop for dy fixnum from 0 below V
                            for dypos fixnum from pos below dend by nwidth
                            for dxend from (mul (plus (1+ y) dy) nwidth) by nwidth do
                            (loop for dx fixnum from 0 below H
                                  for dpos fixnum from dypos below dxend by 3 do
                                  (setf (aref buffer dpos) val))))))))

    (let* ((orig-fn (fdefinition 'upsample)))
      (setf (fdefinition 'upsample) #'upsample-orig)
      (let ((buf-orig (decode-image path)))
        (setf (fdefinition 'upsample) orig-fn)
        (let ((diffs 0))
          (dotimes (i (length buf-current))
            (unless (= (aref buf-current i) (aref buf-orig i))
              (incf diffs)))
          (format t "Differences between current upsample and original upsample: ~D / ~D~%"
                  diffs (length buf-current)))))))

(test-upsamples)
(sb-ext:exit :code 0)
