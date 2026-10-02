(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)
(load "tools/test-layout-b.lisp")

(in-package :jpeg)
(use-package :sb-simd-avx2)

(let* ((buf1 (make-array 64 :element-type 'sint16))
       (buf2 (make-array 64 :element-type 'sint16)))
  (dotimes (i 64)
    (let ((v (- (mod (* (1+ i) 1664525) 256) 128)))
      (setf (aref buf1 i) v
            (aref buf2 i) v)))

  ;; Scalar Pass 1
  (let ((stride 8) (b_base 0) (data buf1))
    (loop with tmp0 fixnum and tmp1 fixnum and tmp2 fixnum
          and tmp3 fixnum and tmp4 fixnum and tmp5 fixnum
          and tmp6 fixnum and tmp7 fixnum and tmp10 fixnum
          and tmp11 fixnum and tmp12 fixnum and tmp13 fixnum
          and z1 fixnum and z2 fixnum and z3 fixnum
          and z4 fixnum and z5 fixnum do
      (loop for r fixnum from 0 to 7
            for row_offset fixnum = (+ b_base (* r stride)) do
        (let ((d0 (+ row_offset 0)) (d1 (+ row_offset 1))
              (d2 (+ row_offset 2)) (d3 (+ row_offset 3))
              (d4 (+ row_offset 4)) (d5 (+ row_offset 5))
              (d6 (+ row_offset 6)) (d7 (+ row_offset 7)))
          (setf tmp0 (+ (aref data d0) (aref data d7)))
          (setf tmp7 (- (aref data d0) (aref data d7)))
          (setf tmp1 (+ (aref data d1) (aref data d6)))
          (setf tmp6 (- (aref data d1) (aref data d6)))
          (setf tmp2 (+ (aref data d2) (aref data d5)))
          (setf tmp5 (- (aref data d2) (aref data d5)))
          (setf tmp3 (+ (aref data d3) (aref data d4)))
          (setf tmp4 (- (aref data d3) (aref data d4)))
          (setf tmp10 (+ tmp0 tmp3))
          (setf tmp13 (- tmp0 tmp3))
          (setf tmp11 (+ tmp1 tmp2))
          (setf tmp12 (- tmp1 tmp2))
          (setf (aref data d0) (ash (+ tmp10 tmp11) 1))
          (setf (aref data d4) (ash (- tmp10 tmp11) 1))
          (setf z1 (mul (+ tmp12 tmp13) +fix-0-541196100+))
          (setf (aref data d2) (descale (+ z1 (mul tmp13 +fix-0-765366865+)) +shift-1+))
          (setf (aref data d6) (descale (+ z1 (mul tmp12 (- +fix-1-847759065+))) +shift-1+))
          (setf z1 (+ tmp4 tmp7))
          (setf z2 (+ tmp5 tmp6))
          (setf z3 (+ tmp4 tmp6))
          (setf z4 (+ tmp5 tmp7))
          (setf z5 (mul (+ z3 z4) +fix-1-175875602+))
          (setf tmp4 (mul tmp4 +fix-0-298631336+))
          (setf tmp5 (mul tmp5 +fix-2-053119869+))
          (setf tmp6 (mul tmp6 +fix-3-072711026+))
          (setf tmp7 (mul tmp7 +fix-1-501321110+))
          (setf z1 (mul z1 (- +fix-0-899976223+)))
          (setf z2 (mul z2 (- +fix-2-562915447+)))
          (setf z3 (mul z3 (- +fix-1-961570560+)))
          (setf z4 (mul z4 (- +fix-0-390180644+)))
          (incf z3 z5)
          (incf z4 z5)
          (setf (aref data d7) (descale (plus3 tmp4 z1 z3) +shift-1+))
          (setf (aref data d5) (descale (plus3 tmp5 z2 z4) +shift-1+))
          (setf (aref data d3) (descale (plus3 tmp6 z2 z3) +shift-1+))
          (setf (aref data d1) (descale (plus3 tmp7 z1 z4) +shift-1+))))
      (return)))

  (format t "Scalar Pass 1 Row 0: ~a~%" (loop for c below 8 collect (aref buf1 c)))

  ;; SIMD Pass 1
  (let* ((r0 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 0)))
         (r1 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 8)))
         (r2 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 16)))
         (r3 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 24)))
         (r4 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 32)))
         (r5 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 40)))
         (r6 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 48)))
         (r7 (s32.8-from-s16.8 (sb-simd-sse2:s16.8-aref buf2 56))))
    (multiple-value-bind (t0 t1 t2 t3 t4 t5 t6 t7)
        (transpose-8x8 r0 r1 r2 r3 r4 r5 r6 r7)
      (multiple-value-bind (p0 p1 p2 p3 p4 p5 p6 p7)
          (vert-dct-p1 t0 t1 t2 t3 t4 t5 t6 t7)
        (format t "SIMD Pass 1 p0 (Col 0 of pass 1): ~a~%" p0)
        (format t "SIMD Pass 1 p1 (Col 1 of pass 1): ~a~%" p1)
        (multiple-value-bind (q0 q1 q2 q3 q4 q5 q6 q7)
            (transpose-8x8 p0 p1 p2 p3 p4 p5 p6 p7)
          (format t "SIMD Pass 1 q0 (Row 0 after transpose): ~a~%" q0))))))

(sb-ext:exit)
