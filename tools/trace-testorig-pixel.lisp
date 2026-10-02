(require :asdf)
(load "~/quicklisp/setup.lisp")
(asdf:load-asd "/home/arthur/repos/cl/cl-jpeg-avx512/cl-jpeg.asd")
(asdf:load-system :cl-jpeg)

(in-package :jpeg)

(let* ((path "/home/arthur/repos/libjpeg-turbo/testimages/testorig.jpg")
       (desc-inter (make-descriptor))
       (desc-planar (make-descriptor)))
  ;; Decode interleaved
  (let ((*use-simd* :on) (*upsample-impl* nil))
    (decode-image path :descriptor desc-inter))
  ;; Decode planar
  (let ((*use-simd* :on) (*upsample-impl* #'planar-upsample))
    (decode-image path :descriptor desc-planar))

  (let* ((w (descriptor-width desc-planar))
         (h (descriptor-height desc-planar))
         (buf-int (descriptor-buffer desc-inter))
         (buf-pln (descriptor-buffer desc-planar))
         (y-pln   (descriptor-y-plane desc-planar))
         (cb-pln  (descriptor-cb-plane desc-planar))
         (cr-pln  (descriptor-cr-plane desc-planar))
         (cw (ash (1+ w) -1)))
    (format t "Image dimensions: ~D x ~D, Chroma W: ~D~%" w h cw)
    
    ;; Find differing pixels on right edge (x=226) and bottom edge (y=148)
    (format t "~%--- Differing pixels on Right Edge (x=226) ---~%")
    (dotimes (y h)
      (let* ((idx (* (+ (* y w) 226) 3))
             (bi (aref buf-int idx)) (gi (aref buf-int (+ idx 1))) (ri (aref buf-int (+ idx 2)))
             (bp (aref buf-pln idx)) (gp (aref buf-pln (+ idx 1))) (rp (aref buf-pln (+ idx 2))))
        (when (or (/= bi bp) (/= gi gp) (/= ri rp))
          (format t "y=~3D: Int BGR=(~D,~D,~D) Pln BGR=(~D,~D,~D) Delta=(~D,~D,~D)~%"
                  y bi gi ri bp gp rp (- bp bi) (- gp gi) (- rp ri)))))

    (format t "~%--- Differing pixels on Bottom Edge (y=148) ---~%")
    (dotimes (x w)
      (let* ((idx (* (+ (* 148 w) x) 3))
             (bi (aref buf-int idx)) (gi (aref buf-int (+ idx 1))) (ri (aref buf-int (+ idx 2)))
             (bp (aref buf-pln idx)) (gp (aref buf-pln (+ idx 1))) (rp (aref buf-pln (+ idx 2))))
        (when (or (/= bi bp) (/= gi gp) (/= ri rp))
          (format t "x=~3D: Int BGR=(~D,~D,~D) Pln BGR=(~D,~D,~D) Delta=(~D,~D,~D)~%"
                  x bi gi ri bp gp rp (- bp bi) (- gp gi) (- rp ri)))))

    ;; Detailed trace for specific pixels:
    ;; 1. Pixel (226, 10) on right edge
    ;; 2. Pixel (224, 10) in boundary strip
    ;; 3. Pixel (148, 148) on bottom edge
    (dolist (pt (list '(226 10) '(224 10) '(220 148)))
      (let* ((px (first pt))
             (py (second pt))
             (idx (* (+ (* py w) px) 3))
             (bi (aref buf-int idx)) (gi (aref buf-int (+ idx 1))) (ri (aref buf-int (+ idx 2)))
             (bp (aref buf-pln idx)) (gp (aref buf-pln (+ idx 1))) (rp (aref buf-pln (+ idx 2)))
             (yy (aref y-pln (+ (* py w) px)))
             (cx (ash px -1))
             (cy (ash py -1))
             (cb (aref cb-pln (+ (* cy cw) cx)))
             (cr (aref cr-pln (+ (* cy cw) cx)))
             ;; Table lookup arithmetic (as done by scalar / tables)
             (cb-g (aref +cb-g-tab+ cb))
             (cr-g (aref +cr-g-tab+ cr))
             (pre-shift (+ cb-g cr-g))
             (g-delta (ash pre-shift (- shift)))
             (pre-clamp-g (+ yy g-delta))
             (final-g (limit pre-clamp-g)))
        (format t "~%--- Detailed Trace for Pixel (~D, ~D) ---~%" px py)
        (format t "  Planar Y: ~D, Cb: ~D (at cx=~D, cy=~D), Cr: ~D~%" yy cb cx cy cr)
        (format t "  Table lookup: cb-g-tab[~D] = ~D, cr-g-tab[~D] = ~D~%" cb cb-g cr cr-g)
        (format t "  pre-shift sum = ~D, ash -shift = ~D, pre-clamp-g = ~D, final-g = ~D~%"
                pre-shift g-delta pre-clamp-g final-g)
        (format t "  Interleaved decoded pixel: B=~D, G=~D, R=~D~%" bi gi ri)
        (format t "  Planar decoded pixel:      B=~D, G=~D, R=~D~%" bp gp rp)
        (format t "  => Interleaved G is ~D, Planar G is ~D (diff = ~D)~%"
                gi gp (- gp gi))))))
