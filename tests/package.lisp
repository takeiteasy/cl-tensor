(defpackage #:cl-tensor/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:ct #:cl-tensor))
  (:export #:run-tests))

(in-package #:cl-tensor/tests)

(def-suite :cl-tensor)

(defun run-tests ()
  (run! :cl-tensor))
