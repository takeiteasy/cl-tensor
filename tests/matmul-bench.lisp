(require :asdf)
(asdf:load-system :cl-tensor)

(defpackage #:cl-tensor/matmul-bench
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor)))

(in-package #:cl-tensor/matmul-bench)

(defun measure (function)
  (dotimes (i 5) (funcall function))
  (let ((iterations 1))
    (loop
      (let ((start (get-internal-real-time)))
        (dotimes (i iterations) (funcall function))
        (when (>= (- (get-internal-real-time) start) (* 0.05 internal-time-units-per-second))
          (return)))
      (setf iterations (* 2 iterations)))
    (let ((times (loop repeat 3 collect
                      (let ((start (get-internal-real-time)))
                        (dotimes (i iterations) (funcall function))
                        (* 1d6 (/ (- (get-internal-real-time) start)
                                  internal-time-units-per-second iterations)))))
          (count (min iterations 100)))
      #+sbcl (sb-ext:gc :full t)
      (let ((before #+sbcl (sb-ext:get-bytes-consed) #-sbcl 0))
        (dotimes (i count) (funcall function))
        (values (second (sort times #'<))
                #+sbcl (/ (- (sb-ext:get-bytes-consed) before) (float count 1d0))
                #-sbcl nil)))))

(defun benchmark-case (dtype case)
  (destructuring-bind (name count rows inner cols rs cs stride) case
    (declare (ignore name))
    (let* ((shape (append (unless (= count 1) (list count)) (list rows cols)))
           (data (ct:tensor-storage
                  (ct:full (list (+ 1 (* (1- count) stride) (* (1- rows) rs) (* (1- cols) cs)))
                           -7 :dtype dtype)))
           (out (ct:make-tensor-view data shape
                                    :strides (append (unless (= count 1) (list stride)) (list rs cs))))
           (a (ct:full (append (unless (= count 1) (list count)) (list rows inner)) 0.5 :dtype dtype))
           (b (ct:full (list inner cols) 0.25 :dtype dtype))
           (written (make-array (length data) :element-type 'bit :initial-element 0)))
      (ct:matmul! out a b)
      (dotimes (batch count)
        (dotimes (row rows)
          (dotimes (col cols)
            (let ((index (+ (* batch stride) (* row rs) (* col cs))))
              (assert (= (* inner 0.125) (aref data index)))
              (setf (aref written index) 1)))))
      (dotimes (index (length data))
        (when (zerop (aref written index)) (assert (= -7 (aref data index)))))
      (lambda () (ct:matmul! out a b)))))

(defun run-benchmarks (label)
  (dolist (backend (if trivial-simd::*native-blas-available-p* '(:lisp :native) '(:lisp)))
    (let ((trivial-simd::*backend* backend)
          (trivial-simd/blas::*native-blas-threshold* 1))
      (dolist (dtype '(:f32 :f64))
        (dolist (case '((matrix-3x2 1 3 4 2 2 3 8)
                        (matrix-9x8 1 9 8 8 8 9 128)
                        (interleaved-matrices 256 3 4 2 2 3 8)
                        (interleaved-batches 256 2 4 2 3 2 4)
                        (contiguous-control 256 3 4 2 2 1 6)))
          (multiple-value-bind (microseconds bytes) (measure (benchmark-case dtype case))
            (format t "~A ~A ~A ~A ~,3F ~A~%" label backend dtype (first case)
                    microseconds (if bytes (format nil "~,0F" bytes) "n/a"))))))))

(defun load-benchmark-source (path)
  (load (compile-file path :output-file
                     (merge-pathnames (make-pathname :name (pathname-name path) :type "fasl")
                                      (uiop:temporary-directory)))))

(let* ((root (merge-pathnames "../" (uiop:pathname-directory-pathname *load-truename*)))
       (baseline (uiop:getenv "CL_TENSOR_MATMUL_BASELINE"))
       (upstream (uiop:getenv "TRIVIAL_SIMD_GEMM_BASELINE")))
  (format t "~&~A ~A ~A~%Implementation Backend Dtype Layout us/call Lisp-bytes/call~%"
          (lisp-implementation-type) (lisp-implementation-version) (machine-type))
  (when (or baseline upstream)
    (unless (and baseline upstream) (error "Both baseline source paths are required"))
    (unwind-protect
         (progn (load-benchmark-source upstream) (load-benchmark-source baseline)
                (run-benchmarks "baseline"))
      (load-benchmark-source (merge-pathnames "blas/level3.lisp"
                                            (asdf:system-source-directory :trivial-simd)))
      (load-benchmark-source (merge-pathnames "matmul.lisp" root))))
  (run-benchmarks "current")
  (dolist (case '((small #(2 10 10) #(1 22 26))
                  (large #(2 1000000 1000000) #(1 2000002 2000006))))
    (destructuring-bind (name dimensions steps) case
      (let ((function (lambda () (trivial-simd/blas::gemm-layout-unique-p dimensions steps))))
        (assert (funcall function))
        (multiple-value-bind (microseconds bytes) (measure function)
          (format t "proof ~A ~,3F us/call ~A Lisp-bytes/call~%"
                  name microseconds (if bytes (format nil "~,0F" bytes) "n/a")))))))
