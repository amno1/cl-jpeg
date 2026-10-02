(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun run-experiment ()
  (let ((path "/tmp/photo3_1024x1024.jpg"))
    ;; Warmup
    (dotimes (i 5)
      (let ((*use-simd* :on) (*upsample-impl* nil)) (decode-image path))
      (let ((*use-simd* :on) (*upsample-impl* #'planar-upsample)) (decode-image path)))
    
    (format t "--- Order A: Planar then Interleaved (50 reps) ---~%")
    (sb-ext:gc :full t)
    (format t "Planar:~%")
    (let ((*use-simd* :on) (*upsample-impl* #'planar-upsample))
      (time (dotimes (i 50) (decode-image path))))
    
    (sb-ext:gc :full t)
    (format t "Interleaved:~%")
    (let ((*use-simd* :on) (*upsample-impl* nil))
      (time (dotimes (i 50) (decode-image path))))

    (format t "~%--- Order B: Interleaved then Planar (50 reps) ---~%")
    (sb-ext:gc :full t)
    (format t "Interleaved:~%")
    (let ((*use-simd* :on) (*upsample-impl* nil))
      (time (dotimes (i 50) (decode-image path))))
    
    (sb-ext:gc :full t)
    (format t "Planar:~%")
    (let ((*use-simd* :on) (*upsample-impl* #'planar-upsample))
      (time (dotimes (i 50) (decode-image path))))))

(run-experiment)
