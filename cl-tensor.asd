(asdf:defsystem "cl-tensor"
  :description "NumPy-like N-dimensional tensors for Common Lisp, built on trivial-simd"
  :author "George Watson"
  :license "MIT"
  :version "0.0.1"
  :depends-on ("trivial-simd" "trivial-simd/blas")
  :serial t
  :components ((:file "package") (:file "dtype") (:file "tensor"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-tensor/tests"))))

(asdf:defsystem "cl-tensor/tests"
  :depends-on ("cl-tensor" "fiveam")
  :serial t
  :components ((:file "tests/package") (:file "tests/tensor") (:file "tests/random"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-tensor/tests :run-tests)
               (error "cl-tensor tests failed"))))
