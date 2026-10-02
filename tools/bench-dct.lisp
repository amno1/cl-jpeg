(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(defun bench-scalar-dct (&key (nblocks 65536) (reps 10))
  (let ((buf (make-array (* nblocks 64) :element-type 'sint16 :initial-element 100)))
    (declare (type (simple-array sint16 (*)) buf))
    (format t "Benchmarking scalar llm-dct on ~d blocks (~,2f MPix)...~%" nblocks (/ (* nblocks 64) 1000000.0))
    ;; Warmup
    (loop for b fixnum from 0 below (min nblocks 1024) do
      (llm-dct buf b nblocks))
    (let ((times nil))
      (dotimes (r reps)
        (let ((t0 (get-internal-real-time)))
          (loop for b fixnum from 0 below nblocks do
            (llm-dct buf b nblocks))
          (let* ((t1 (get-internal-real-time))
                 (dt (/ (- t1 t0) (float internal-time-units-per-second))))
            (push dt times)
            (format t "  Rep ~2d: ~,2f ms | ~,1f MPix/s | ~,1f ns/block~%"
                    r (* dt 1000.0) (/ (* nblocks 64) dt 1000000.0) (/ (* dt 1000000000.0) nblocks)))))
      (let* ((mean (/ (reduce #'+ times) (length times)))
             (variance (/ (reduce #'+ (mapcar (lambda (x) (expt (- x mean) 2)) times)) (length times)))
             (std (sqrt variance)))
        (format t "SUMMARY: Mean: ~,2f ms +/- ~,2f ms | ~,1f MPix/s | ~,1f ns/block~%"
                (* mean 1000.0) (* std 1000.0) (/ (* nblocks 64) mean 1000000.0) (/ (* mean 1000000000.0) nblocks))))))

(bench-scalar-dct)
(sb-ext:exit)
