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
         (y-pln  (descriptor-y-plane desc-planar))
         (cb-pln (descriptor-cb-plane desc-planar))
         (cr-pln (descriptor-cr-plane desc-planar))
         (cw (ash (1+ w) -1))
         (diff-count 0)
         (diff-pts nil))
    (format t "Image dimensions: ~D x ~D (~D pixels)~%" w h (* w h))
    (dotimes (y h)
      (dotimes (x w)
        (let* ((linear-idx (+ (* y w) x))
               (byte-idx (* linear-idx 3))
               (bs (aref buf-s byte-idx)) (gs (aref buf-s (+ byte-idx 1))) (rs (aref buf-s (+ byte-idx 2)))
               (bi (aref buf-i byte-idx)) (gi (aref buf-i (+ byte-idx 1))) (ri (aref buf-i (+ byte-idx 2)))
               (bp (aref buf-p byte-idx)) (gp (aref buf-p (+ byte-idx 1))) (rp (aref buf-p (+ byte-idx 2))))
          (when (or (/= bi bp) (/= gi gp) (/= ri rp))
            (incf diff-count)
            (push (list x y linear-idx
                        (list bs gs rs)
                        (list bi gi ri)
                        (list bp gp rp))
                  diff-pts)))))
    (setf diff-pts (nreverse diff-pts))
    (format t "Total diverging pixels between Interleaved and Planar: ~D~%" diff-count)
    
    ;; Compare each against scalar
    (let ((pln-matches-scalar 0)
          (int-matches-scalar 0)
          (both-differ-from-scalar 0))
      (dolist (pt diff-pts)
        (destructuring-bind (x y lidx s-bgr i-bgr p-bgr) pt
          (declare (ignore x y lidx))
          (cond
            ((equal p-bgr s-bgr) (incf pln-matches-scalar))
            ((equal i-bgr s-bgr) (incf int-matches-scalar))
            (t (incf both-differ-from-scalar)))))
      (format t "Of ~D diverging pixels:~%" diff-count)
      (format t "  Planar matches Scalar bit-exact:       ~D (~,2F%)~%"
              pln-matches-scalar (* 100.0 (/ pln-matches-scalar diff-count)))
      (format t "  Interleaved matches Scalar bit-exact:  ~D (~,2F%)~%"
              int-matches-scalar (* 100.0 (/ int-matches-scalar diff-count)))
      (format t "  Both differ from Scalar:               ~D~%" both-differ-from-scalar))

    ;; Print breakdown of coordinates
    (format t "~%All ~D diverging pixels (x, y, linear-idx):~%" diff-count)
    (dolist (pt diff-pts)
      (destructuring-bind (x y lidx s-bgr i-bgr p-bgr) pt
        (format t "  (~3D, ~3D) lidx=~5D: Scal=~A Int=~A Pln=~A~%"
                x y lidx s-bgr i-bgr p-bgr)))

    ;; Detailed analysis of selected pixels
    (dolist (pt (list '(226 10) '(224 10) '(220 148) '(226 148)))
      (let* ((px (first pt))
             (py (second pt))
             (lidx (+ (* py w) px))
             (bidx (* lidx 3))
             (bs (aref buf-s bidx)) (gs (aref buf-s (+ bidx 1))) (rs (aref buf-s (+ bidx 2)))
             (bi (aref buf-i bidx)) (gi (aref buf-i (+ bidx 1))) (ri (aref buf-i (+ bidx 2)))
             (bp (aref buf-p bidx)) (gp (aref buf-p (+ bidx 1))) (rp (aref buf-p (+ bidx 2)))
             (cx (ash px -1))
             (cy (ash py -1))
             (yy (aref y-pln (+ (* py w) px)))
             (cb (aref cb-pln (+ (* cy cw) cx)))
             (cr (aref cr-pln (+ (* cy cw) cx)))
             ;; Scalar table arithmetic:
             (cb-g-val (aref +cb-g-tab+ cb))
             (cr-g-val (aref +cr-g-tab+ cr))
             (sum-g (+ cb-g-val cr-g-val))
             (g-shift (ash sum-g (- shift)))
             (pre-clamp-scalar-g (+ yy g-shift))
             (scalar-g (limit pre-clamp-scalar-g))
             ;; SIMD vpmulhrsw arithmetic:
             (cb-prime (- cb 128))
             (cr-prime (- cr 128))
             (cb-term (ash (+ (* cb-prime +ycc-cb-g+) 16384) -15))
             (cr-term (ash (+ (* cr-prime +ycc-cr-g+) 16384) -15))
             (pre-clamp-simd-g (+ yy cb-term cr-term))
             (simd-g (limit pre-clamp-simd-g)))
        (format t "~%================================================================~%")
        (format t "Detailed Analysis of Pixel (~D, ~D) [linear index ~D]:~%" px py lidx)
        (format t "  Chroma coordinates: cx=~D (px>>1), cy=~D (py>>1), chroma stride=~D~%"
                cx cy cw)
        (format t "  Y = ~D, Cb = ~D, Cr = ~D~%" yy cb cr)
        (format t "  Decoded values: Scalar=(~D,~D,~D) Interleaved=(~D,~D,~D) Planar=(~D,~D,~D)~%"
                bs gs rs bi gi ri bp gp rp)
        (format t "  Scalar table arithmetic:~%")
        (format t "    +cb-g-tab+[~D] = ~D~%" cb cb-g-val)
        (format t "    +cr-g-tab+[~D] = ~D~%" cr cr-g-val)
        (format t "    sum = ~D~%" sum-g)
        (format t "    ash sum -~D = ~D~%" shift g-shift)
        (format t "    pre-clamp green = ~D + ~D = ~D~%" yy g-shift pre-clamp-scalar-g)
        (format t "    final clamped green = ~D~%" scalar-g)
        (format t "  SIMD vpmulhrsw arithmetic:~%")
        (format t "    Cb' = ~D - 128 = ~D,  Cr' = ~D - 128 = ~D~%" cb cb-prime cr cr-prime)
        (format t "    Cb term = (Cb' * ~D + 16384) >> 15 = (~D + 16384) >> 15 = ~D~%"
                +ycc-cb-g+ (* cb-prime +ycc-cb-g+) cb-term)
        (format t "    Cr term = (Cr' * ~D + 16384) >> 15 = (~D + 16384) >> 15 = ~D~%"
                +ycc-cr-g+ (* cr-prime +ycc-cr-g+) cr-term)
        (format t "    pre-clamp green = ~D + ~D + ~D = ~D~%"
                yy cb-term cr-term pre-clamp-simd-g)
        (format t "    final clamped green = ~D~%" simd-g)
        (format t "  Which path did Interleaved execute? ~A~%"
                (if (< lidx (* (floor (* w h) 16) 16)) "SIMD (vpmulhrsw)" "Scalar Tail (table lookup)"))
        (format t "  Which path did Planar execute?      ~A~%"
                (if (< px (* (floor w 32) 32)) "SIMD (vpmulhrsw)" "Scalar Tail (table lookup)"))))))
