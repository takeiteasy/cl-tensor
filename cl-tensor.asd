(asdf:defsystem "cl-tensor"
  :description "NumPy-like N-dimensional tensors for Common Lisp, built on trivial-simd"
  :author "George Watson"
  :license "MIT"
  :version "0.0.1"
  :depends-on ("trivial-simd" "trivial-simd/blas")
  :serial t
  :components ((:file "package")))
