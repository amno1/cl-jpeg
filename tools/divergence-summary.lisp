(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(let* ((path "/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg")
       (desc-scalar  (make-descriptor))
       (desc-inter   (make-descriptor))
       (desc-planar  (make-descriptor)))
  ;; 1. Scalar decode
  (let ((*use-simd* :off) (*upsample-impl* nil))
    (decode-image path :descriptor desc-scalar))
  ;; 2. Interleaved SIMD decode
  (let ((*use-simd* :on) (*upsample-impl* nil))
    (decode-image path :descriptor desc-inter))
  ;; 3. Planar SIMD decode
  (let ((*use-simd* :on) (*upsample-impl* #'planar-upsample))
    (decode-image path :descriptor desc-planar))

  (let* ((w (descriptor-width desc-scalar))
         (h (descriptor-height desc-scalar))
         (buf-s (descriptor-buffer desc-scalar))
         (buf-i (descriptor-buffer desc-inter))
         (buf-p (descriptor-buffer desc-planar))
         (diff-count 0)
         (pln-matches-scalar 0)
         (int-matches-scalar 0)
         (both-differ 0)
         (channel-diffs (list 0 0 0))) ; B G R
    (dotimes (y h)
      (dotimes (x w)
        (let* ((lidx (+ (* y w) x))
               (bidx (* lidx 3))
               (bs (aref buf-s bidx)) (gs (aref buf-s (+ bidx 1))) (rs (aref buf-s (+ bidx 2)))
               (bi (aref buf-i bidx)) (gi (aref buf-i (+ bidx 1))) (ri (aref buf-i (+ bidx 2)))
               (bp (aref buf-p bidx)) (gp (aref buf-p (+ bidx 1))) (rp (aref buf-p (+ bidx 2))))
          (when (or (/= bi bp) (/= gi gp) (/= ri rp))
            (incf diff-count)
            (when (/= bi bp) (incf (nth 0 channel-diffs)))
            (when (/= gi gp) (incf (nth 1 channel-diffs)))
            (when (/= ri rp) (incf (nth 2 channel-diffs)))
            (cond
              ((and (= bp bs) (= gp gs) (= rp rs)) (incf pln-matches-scalar))
              ((and (= bi bs) (= gi gs) (= ri rs)) (incf int-matches-scalar))
              (t (incf both-differ)))))))
    (format t "Total diverging pixels: ~D / ~D (~,4F%)~%"
            diff-count (* w h) (* 100.0 (/ diff-count (* w h))))
    (format t "Channel breakdown of differences (B, G, R): ~A~%" channel-diffs)
    (format t "  Planar matches Reference Scalar bit-exact:       ~D (~,2F%)~%"
            pln-matches-scalar (* 100.0 (/ pln-matches-scalar diff-count)))
    (format t "  Interleaved matches Reference Scalar bit-exact:  ~D (~,2F%)~%"
            int-matches-scalar (* 100.0 (/ int-matches-scalar diff-count)))
    (format t "  Both differ from Reference Scalar:               ~D~%" both-differ)))
