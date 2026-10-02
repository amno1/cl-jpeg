(asdf:defsystem :cl-jpeg
  :name "cl-jpeg"
  :version "3.0.0"
  :license "BSD"
  :description "A self-contained baseline JPEG codec implementation"
  :author "Eugene Zaikonnikov; contributions by Kenan Bölükbaşı, Manuel Giraud, Cyrus Harmon and William Halliburton"
  :components ((:file "package")
	       (:file "jpeg" :depends-on ("package"))
	       #+(and sbcl x86-64) (:file "simd" :depends-on ("jpeg"))
	       (:file "io" :depends-on ("jpeg"))))
