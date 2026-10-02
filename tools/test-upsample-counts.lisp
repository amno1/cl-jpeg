(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)
(defvar *count-1x1* 0)
(defvar *count-2x2* 0)
(defvar *count-fallback* 0)

(defun upsample (image scan block x y H V offset nwidth nw nx dend)
  (declare #.*optimize*
           (type sint16-2d-array block)
           (type fixnum x y H V offset nwidth nw nx dend))
  (let* ((buffer (descriptor-buffer image))
         (xbase (+ (scan-x scan) x))
         (ybase (+ (scan-y scan) y))
         (img-w (descriptor-width image))
         (img-h (descriptor-height image))
         (ncomp (descriptor-ncomp image))
         (base-pos (+ (* ybase nwidth) (* xbase ncomp))))
    (declare (type uint8-array buffer)
             (type fixnum xbase ybase img-w img-h ncomp base-pos))
    (cond
      ((and (= H 1) (= V 1) (= ncomp 3) (<= (+ xbase 8) img-w) (<= (+ ybase 8) img-h))
       (incf *count-1x1*)
       (upsample-1x1-fast buffer block offset base-pos nwidth))
      ((and (= H 2) (= V 2) (= ncomp 3) (<= (+ xbase 16) img-w) (<= (+ ybase 16) img-h))
       (incf *count-2x2*)
       (upsample-2x2-fast buffer block offset base-pos nwidth))
      (t
       (incf *count-fallback*)
       (let* ((nxbase (* xbase ncomp))
              (nybase (* ybase nwidth)))
         (declare (type fixnum nxbase nybase))
         (loop for row of-type sint16-array across block
               for y-idx fixnum from ybase below (descriptor-height image) by V
               for ypos fixnum from nybase by nw do
               (loop for val fixnum across row
                     for x-idx fixnum from xbase below (descriptor-width image) by H
                     for pos fixnum from (the fixnum (+ ypos offset nxbase)) by nx do
                     (if (= 1 H V)
                         (setf (aref buffer pos) val)
                         (loop for dy fixnum from 0 below V
                               for dypos fixnum from pos below dend by nwidth
                               for dxend from (* (+ (1+ y-idx) dy) nwidth) by nwidth do
                               (loop for dx fixnum from 0 below H
                                     for dpos fixnum from dypos below dxend by 3 do
                                     (setf (aref buffer dpos) val)))))))))))

(setf *count-1x1* 0 *count-2x2* 0 *count-fallback* 0)
(decode-image "/tmp/photo1_512x512.jpg")
(format t "Photo1 (512x512):   1x1=~D, 2x2=~D, fallback=~D~%" *count-1x1* *count-2x2* *count-fallback*)

(setf *count-1x1* 0 *count-2x2* 0 *count-fallback* 0)
(decode-image "/tmp/photo3_1024x1024.jpg")
(format t "Photo3 (1024x1024): 1x1=~D, 2x2=~D, fallback=~D~%" *count-1x1* *count-2x2* *count-fallback*)

(sb-ext:exit :code 0)
