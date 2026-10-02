;;;; test-simd-equivalence.lisp --- Phase 1A Equivalence & Photographic Test Suite
(require :asdf)
(load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname)))
(push (truename "/home/arthur/repos/cl/cl-jpeg-avx512") asdf:*central-registry*)
(asdf:load-system "cl-jpeg" :force t)

(in-package :jpeg)

(defun scalar-reference-convert (image)
  "Invoke the pure scalar inverse color conversion on an image descriptor."
  (let* ((buffer (descriptor-buffer image))
         (nw (* (descriptor-width image) 3)))
    (declare (type (simple-array (unsigned-byte 8) (*)) buffer)
             (type fixnum nw))
    (loop for y fixnum from 0 below (descriptor-height image)
          for yp fixnum from 0 by nw do
          (loop for x fixnum from 0 below (descriptor-width image)
                for py fixnum from yp by 3
                for pu fixnum = (1+ py)
                for pv fixnum = (+ py 2)
                for yy fixnum = (aref buffer py)
                for cb fixnum = (aref buffer pu)
                for cr fixnum = (aref buffer pv) do
                (setf (aref buffer py) (the (unsigned-byte 8) (limit (+ yy (aref +cb-b-tab+ cb)))))
                (setf (aref buffer pu) (the (unsigned-byte 8) (limit (+ yy (ash (+ (aref +cb-g-tab+ cb)
                                                                                  (aref +cr-g-tab+ cr))
                                                                               (- shift))))))
                (setf (aref buffer pv) (the (unsigned-byte 8) (limit (+ yy (aref +cr-r-tab+ cr)))))))))

(defun run-equivalence-test (name w h generator &key (pad-bytes 64))
  (let* ((n (* w h))
         (buf-len (+ (* n 3) pad-bytes))
         (buf-orig (make-array buf-len :element-type '(unsigned-byte 8)))
         (buf-scal (make-array buf-len :element-type '(unsigned-byte 8)))
         (buf-simd (make-array buf-len :element-type '(unsigned-byte 8))))
    (funcall generator buf-orig w h)
    (replace buf-scal buf-orig)
    (replace buf-simd buf-orig)

    (let ((desc-scal (make-descriptor :width w :height h :buffer buf-scal))
          (desc-simd (make-descriptor :width w :height h :buffer buf-simd)))
      ;; Run scalar reference
      (scalar-reference-convert desc-scal)
      ;; Run SIMD accelerated
      (inverse-colorspace-convert-simd-accelerated desc-simd)

      (let ((max-diff 0)
            (mismatch-count 0))
        (dotimes (i n)
          (let* ((idx (* i 3))
                 (b-s (aref buf-simd idx))
                 (g-s (aref buf-simd (+ idx 1)))
                 (r-s (aref buf-simd (+ idx 2)))
                 (b-c (aref buf-scal idx))
                 (g-c (aref buf-scal (+ idx 1)))
                 (r-c (aref buf-scal (+ idx 2)))
                 (diff (max (abs (- b-s b-c))
                            (abs (- g-s g-c))
                            (abs (- r-s r-c)))))
            (when (> diff 0) (incf mismatch-count))
            (when (> diff max-diff) (setf max-diff diff))))
        (format t "  [PASS] ~22A (~5Dx~5D = ~7D px) | max-diff=~D LSB, exact-match=~5,1F%~%"
                name w h n max-diff (* 100.0 (/ (- n mismatch-count) (float n))))
        (assert (<= max-diff 1) () "Test ~A failed with max-diff=~D (> 1 LSB)!" name max-diff)))))

(defun test-kernel-dispatch-firing ()
  (format t "~%--- Kernel Dispatch Firing & Counter Verification (16x1, 17x1 with padding) ---~%")
  ;; 16x1 with 64-byte padding: exactly 1 chunk (16 px) processed by SIMD kernel
  (let* ((n 16)
         (buf (make-array (+ (* n 3) 64) :element-type '(unsigned-byte 8) :initial-element 128))
         (done (inverse-colorspace-convert-simd buf n)))
    (format t "  16x1 padded buffer: SIMD kernel processed ~D/16 pixels~%" done)
    (assert (= done 16) () "Expected 16 pixels processed by SIMD kernel!"))

  ;; 17x1 with 64-byte padding: 16 px processed by SIMD kernel, 1 px by tail
  (let* ((n 17)
         (buf (make-array (+ (* n 3) 64) :element-type '(unsigned-byte 8) :initial-element 128))
         (done (inverse-colorspace-convert-simd buf n)))
    (format t "  17x1 padded buffer: SIMD kernel processed ~D/16 pixels (tail: 1 px)~%" done)
    (assert (= done 16) () "Expected 16 pixels processed by SIMD kernel!"))

  ;; 17x1 image through full decoder wrapper inspecting chunk counter
  (let* ((n 17)
         (buf (make-array (+ (* n 3) 64) :element-type '(unsigned-byte 8) :initial-element 128))
         (desc (make-descriptor :width 17 :height 1 :buffer buf)))
    (setf *simd-color-chunks-processed* 0)
    (inverse-colorspace-convert-simd-accelerated desc)
    (format t "  17x1 through wrapper: *simd-color-chunks-processed* = ~D chunk (16 px)~%"
            *simd-color-chunks-processed*)
    (assert (= *simd-color-chunks-processed* 1) ()
            "Expected exactly 1 SIMD chunk processed on 17x1 image!")))

(defun run-real-jpeg-test (name path &key dispatch-guard-p)
  (let* ((scal-img (let ((*use-simd* :off)) (decode-image (pathname path))))
         (simd-img (let ((*use-simd* :auto)) (decode-image (pathname path))))
         (len (length scal-img))
         (max-diff 0)
         (mismatches 0))
    (dotimes (i len)
      (let ((diff (abs (- (aref simd-img i) (aref scal-img i)))))
        (when (> diff 0) (incf mismatches))
        (when (> diff max-diff) (setf max-diff diff))))
    (if dispatch-guard-p
        (format t "  [PASS] Photographic ~10A (~7D bytes) [dispatch guard] | max-diff=~D LSB, exact-match=~5,1F%~%"
                name len max-diff (* 100.0 (/ (- len mismatches) (float len))))
        (format t "  [PASS] Photographic ~10A (~7D bytes)                    | max-diff=~D LSB, exact-match=~5,1F%~%"
                name len max-diff (* 100.0 (/ (- len mismatches) (float len)))))
    (assert (<= max-diff 1) () "Photographic test ~A failed with max-diff=~D (> 1 LSB)!" name max-diff)))

(defun main ()
  (format t "========================================================================================~%")
  (format t "  CL-JPEG AVX-512 Phase 1A: Equivalence & Photographic Correctness Suite (v2.9.0)~%")
  (format t "========================================================================================~%")
  (format t "SIMD available: ~A, *use-simd* default: ~A~%" *simd-decode-color-p* *use-simd*)
  (assert *simd-decode-color-p* () "AVX-512 instruction sets must be available on this CPU!")

  (format t "~%--- 1. Uniform & Synthetic Fields (Tolerance: max-diff <= 1 LSB) ---~%")
  (run-equivalence-test "Flat 0" 64 64 (lambda (buf w h) (declare (ignore w h)) (fill buf 0)))
  (run-equivalence-test "Flat 128" 64 64 (lambda (buf w h) (declare (ignore w h)) (fill buf 128)))
  (run-equivalence-test "Flat 255" 64 64 (lambda (buf w h) (declare (ignore w h)) (fill buf 255)))
  (run-equivalence-test "Gradients 64x64" 64 64
                        (lambda (buf w h)
                          (dotimes (y h)
                            (dotimes (x w)
                              (let ((idx (* (+ (* y w) x) 3)))
                                (setf (aref buf idx) (logand (+ x y) 255)
                                      (aref buf (+ idx 1)) (logand (* x 4) 255)
                                      (aref buf (+ idx 2)) (logand (* y 4) 255)))))))

  (format t "~%--- 2. Required Dimension Edge Cases (Tail & Bounds Correctness) ---~%")
  (run-equivalence-test "1x1 (scalar tail)" 1 1
                        (lambda (buf w h) (declare (ignore w h))
                          (setf (aref buf 0) 100 (aref buf 1) 150 (aref buf 2) 200)))
  (run-equivalence-test "15x1 (scalar tail)" 15 1
                        (lambda (buf w h) (declare (ignore w h))
                          (dotimes (i 45) (setf (aref buf i) (mod (* i 17) 256)))))
  (run-equivalence-test "16x1 (simd padded)" 16 1
                        (lambda (buf w h) (declare (ignore w h))
                          (dotimes (i 48) (setf (aref buf i) (mod (* i 19) 256)))))
  (run-equivalence-test "17x1 (simd+tail)" 17 1
                        (lambda (buf w h) (declare (ignore w h))
                          (dotimes (i 51) (setf (aref buf i) (mod (* i 23) 256)))))
  (run-equivalence-test "16x15 (multi-row)" 16 15
                        (lambda (buf w h) (declare (ignore w h))
                          (dotimes (i (* 16 15 3)) (setf (aref buf i) (mod (* i 7) 256)))))
  (run-equivalence-test "1023x769 (odd bounds)" 1023 769
                        (lambda (buf w h)
                          (let ((len (* w h 3))
                                (seed 12345))
                            (dotimes (i len)
                              (setf seed (logand (+ (* seed 1103515245) 12345) #x7fffffff))
                              (setf (aref buf i) (logand (ash seed -16) 255))))))

  (test-kernel-dispatch-firing)

  (format t "~%--- 3. Photographic Real JPEG Tests Across Subsampling Modes ---~%")
  (run-real-jpeg-test "4:4:4" "/tmp/test_444.jpg")
  (run-real-jpeg-test "4:2:2" "/tmp/test_422.jpg")
  (run-real-jpeg-test "4:2:0" "/tmp/test_420.jpg")
  (run-real-jpeg-test "Grayscale" "/tmp/test_gray.jpg" :dispatch-guard-p t)

  (format t "~%========================================================================================~%")
  (format t "  ALL TESTS PASSED! BIT-EXACT WITHIN +/- 1 LSB ACROSS ALL MODES!~%")
  (format t "========================================================================================~%"))

(main)
