
(require :asdf)
(push "/home/arthur/repos/cl/cl-jpeg-avx512/" asdf:*central-registry*)
(asdf:load-system :cl-jpeg)

(in-package :jpeg)
(declaim (type (or null function) *llm-dct-impl*))
(defvar *llm-dct-impl* nil)

(defun test-crunch (buf pos nblocks table)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum pos nblocks)
           (type (simple-array sint16 (*)) buf))
  (if (and (not (eql *use-simd* :off)) *llm-dct-impl*)
      (funcall (the function *llm-dct-impl*) buf pos nblocks)
      (llm-dct buf pos nblocks))
  (quantize buf pos nblocks table))

(disassemble 'test-crunch)
(sb-ext:exit)
