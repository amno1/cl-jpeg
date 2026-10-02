(in-package :cl-user)

(deftype sint16 () '(signed-byte 16))
(deftype sint16-array () '(simple-array sint16 (*)))
(deftype sint16-2d-array () '(simple-array sint16-array (*)))

(defmacro s16ref (data x y)
  `(the sint16 (aref (the sint16-array (aref (the sint16-2d-array ,data) ,y)) ,x)))

;; Test 1: User's exact benchmark with random
(defun test-nested-random (n)
  (declare (type fixnum n) (optimize (speed 3) (safety 0)))
  (let ((a (make-array 8 :element-type 'sint16-array
                         :initial-contents (loop repeat 8 collect (make-array 8 :element-type 'sint16)))))
    (declare (type sint16-2d-array a))
    (dotimes (i n)
      (setf (s16ref a (random 8) (random 8)) (logand i #x7fff)))))

(defun test-flat-random (n)
  (declare (type fixnum n) (optimize (speed 3) (safety 0)))
  (let ((a (make-array 64 :element-type 'sint16)))
    (declare (type (simple-array sint16 (64)) a))
    (dotimes (i n)
      (setf (aref a (+ (random 8) (* 8 (random 8)))) (logand i #x7fff)))))

;; Test 2: Sequential deterministic access
(defun test-nested-det (n)
  (declare (type fixnum n) (optimize (speed 3) (safety 0)))
  (let ((a (make-array 8 :element-type 'sint16-array
                         :initial-contents (loop repeat 8 collect (make-array 8 :element-type 'sint16)))))
    (declare (type sint16-2d-array a))
    (dotimes (i n)
      (let ((x (logand i 7))
            (y (logand (ash i -3) 7)))
        (setf (s16ref a x y) (logand i #x7fff))))))

(defun test-flat-det (n)
  (declare (type fixnum n) (optimize (speed 3) (safety 0)))
  (let ((a (make-array 64 :element-type 'sint16)))
    (declare (type (simple-array sint16 (64)) a))
    (dotimes (i n)
      (let ((x (logand i 7))
            (y (logand (ash i -3) 7)))
        (setf (aref a (the fixnum (+ x (the fixnum (ash y 3))))) (logand i #x7fff))))))

;; Test 3: Untyped row iteration (as in current llm-dct and quantize-block)
(defun test-nested-untyped-loop (reps)
  (declare (type fixnum reps) (optimize (speed 3) (safety 0)))
  (let ((block (make-array 8 :element-type 'sint16-array
                             :initial-contents (loop repeat 8 collect (make-array 8 :element-type 'sint16)))))
    (declare (type sint16-2d-array block))
    (dotimes (r reps)
      (let ((val (logand r #x7fff)))
        (loop for row across block do
              (dotimes (x 8)
                (setf (aref row x) val)))))))

(defun test-flat-typed-loop (reps)
  (declare (type fixnum reps) (optimize (speed 3) (safety 0)))
  (let ((block (make-array 64 :element-type 'sint16)))
    (declare (type (simple-array sint16 (64)) block))
    (dotimes (r reps)
      (let ((val (logand r #x7fff)))
        (dotimes (y 8)
          (let ((yoff (ash y 3)))
            (dotimes (x 8)
              (setf (aref block (+ yoff x)) val))))))))

(defun main ()
  (format t "====================================================================~%")
  (format t "  CONTROL MICRO-BENCHMARK: sint16-2d-array vs Flat Array~%")
  (format t "====================================================================~%")

  (format t "~%--- Experiment 1: User's Random Access (10,000,000 iterations) ---~%")
  (format t "A (nested):~%")
  (time (test-nested-random 10000000))
  (format t "B (flat):~%")
  (time (test-flat-random 10000000))

  (format t "~%--- Experiment 2: Deterministic 1D index (100,000,000 iterations) ---~%")
  (format t "A (nested):~%")
  (time (test-nested-det 100000000))
  (format t "B (flat):~%")
  (time (test-flat-det 100000000))

  (format t "~%--- Experiment 3: Block Sweeps (Untyped Nested vs Flat: 10,000,000 blocks = 640M pixels) ---~%")
  (format t "A (untyped row loop as in current encoder):~%")
  (time (test-nested-untyped-loop 10000000))
  (format t "B (flat 1D indexed loop):~%")
  (time (test-flat-typed-loop 10000000)))

(main)
