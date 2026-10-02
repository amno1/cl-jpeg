(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(load "/home/arthur/repos/cl/cl-jpeg-avx512/tools/test-layout-b.lisp")

(in-package :jpeg)

(declaim (type (or null function) *llm-dct-impl*))
(defvar *llm-dct-impl* #'simd-dct-block-layout-b)

(defun llm-dct-dispatch (data b nblocks)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum b nblocks)
           (type (simple-array sint16 (*)) data))
  (if *llm-dct-impl*
      (funcall (the function *llm-dct-impl*) data b nblocks)
      (llm-dct data b nblocks)))

(declaim (inline crunch))
(defun crunch (buf pos nblocks table)
  (declare #.*optimize*
           (type fixnum pos nblocks)
           (type (simple-array sint16 (*)) buf))
  (llm-dct-dispatch buf pos nblocks)
  (quantize buf pos nblocks table))

;; Recompile encode-stream
(compile 'encode-image-stream)

(defun bench-encoder (&key (reps 20))
  (let* ((width 2048)
         (height 2048)
         (npixels (* width height))
         (image (make-array (* npixels 3) :element-type 'uint8))
         (out-stream (make-broadcast-stream)))
    (declare (type (simple-array uint8 (*)) image))
    ;; Initialize smooth gradient image
    (dotimes (y height)
      (dotimes (x width)
        (let ((idx (* (+ (* y width) x) 3)))
          (setf (aref image idx) (mod x 256)
                (aref image (+ idx 1)) (mod y 256)
                (aref image (+ idx 2)) (mod (+ x y) 256)))))
    ;; Warmup
    (dotimes (i 3)
      (encode-image-stream out-stream image 3 height width :sampling '((2 2) (1 1) (1 1))))
    ;; Measure
    (format t "Benchmarking full 2048x2048 encode (with RECOMPILED SIMD DCT, 20 reps)...~%")
    (let ((times nil))
      (dotimes (r reps)
        (let ((t0 (get-internal-real-time)))
          (encode-image-stream out-stream image 3 height width :sampling '((2 2) (1 1) (1 1)))
          (let* ((t1 (get-internal-real-time))
                 (dt (/ (- t1 t0) (float internal-time-units-per-second))))
            (push dt times)
            (format t "  Rep ~2d: ~,2f ms (~,1f MPix/s)~%" r (* dt 1000.0) (/ npixels dt 1000000.0)))))
      (let* ((mean (/ (reduce #'+ times) (length times)))
             (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) times)) (length times)))
             (std (sqrt variance))
             (mpix (/ npixels mean 1000000.0)))
        (format t "====================================================================~%")
        (format t "FULL ENCODER RESULT WITH RECOMPILED SIMD DCT:~%")
        (format t "  Mean: ~,2f ms +/- ~,2f ms | ~,1f MPix/s~%" (* mean 1000.0) (* std 1000.0) mpix)
        (format t "====================================================================~%")))))

(bench-encoder)
(sb-ext:exit)
