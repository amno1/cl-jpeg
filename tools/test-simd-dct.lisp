(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(require :sb-simd)

(in-package :jpeg)
(use-package :sb-simd-avx2)

(defun test-full-dct-equivalence ()
  (let* ((nblocks 16)
         (stride (* nblocks 8))
         (buf1 (make-array (* nblocks 64) :element-type 'sint16))
         (buf2 (make-array (* nblocks 64) :element-type 'sint16)))
    ;; Fill with pseudo-random and patterned data
    (dotimes (i (* nblocks 64))
      (let ((val (- (mod (* (1+ i) 1103515245) 256) 128)))
        (setf (aref buf1 i) val
              (aref buf2 i) val)))

    ;; 1. Run scalar llm-dct on buf1
    (dotimes (b nblocks)
      (llm-dct buf1 b nblocks))

    ;; 2. Run SIMD 2D DCT with transpose on buf2
    (dotimes (b nblocks)
      (let* ((b_base (ash b 3))
             ;; Load 8 rows as s32.8
             (r0 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 (+ b_base (* 0 stride))))))
        ))
    (format t "Test completed.~%")))

(format t "Compiling test...~%")
(sb-ext:exit)
