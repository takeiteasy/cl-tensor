(defpackage #:cl-tensor/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:ct #:cl-tensor))
  (:export #:run-tests))

(in-package #:cl-tensor/tests)

(def-suite :cl-tensor)

(defun run-tests ()
  (run! :cl-tensor))

(defun all-indices (shape)
  "Every index list of SHAPE in row-major order."
  (if (null shape)
      (list '())
      (loop for i below (first shape)
            nconc (mapcar (lambda (rest) (cons i rest)) (all-indices (rest shape))))))

(defun tensor-values (tensor)
  (mapcar (lambda (indices) (apply #'ct:tref tensor indices))
          (all-indices (coerce (ct:tensor-shape tensor) 'list))))
