(in-package #:cl-tensor/tests)

(in-suite :cl-tensor)

(test dtype-inference
  (is (eq :f32 (ct:storage-dtype (make-array 2 :element-type 'single-float))))
  (is (eq :f64 (ct:storage-dtype (make-array 2 :element-type 'double-float))))
  (is (eq :c32 (ct:storage-dtype (make-array 2 :element-type '(complex single-float)))))
  (is (eq :s8 (ct:storage-dtype (make-array 2 :element-type '(signed-byte 8)))))
  (is (eq :u16 (ct:storage-dtype (make-array 2 :element-type '(unsigned-byte 16)))))
  (signals error (ct:storage-dtype (make-array 2))))

(test sixteen-bit-float-dtypes-are-explicit
  (let ((storage (make-array 4 :element-type '(unsigned-byte 16))))
    (is (eq :u16 (ct:tensor-dtype (ct:make-tensor-view storage '(4)))))
    (is (eq :f16 (ct:tensor-dtype (ct:make-tensor-view storage '(4) :dtype :f16))))
    (is (eq :bf16 (ct:tensor-dtype (ct:make-tensor-view storage '(4) :dtype :bf16))))
    (is (ct:dtype-storage-only-p :f16))
    (is (not (ct:dtype-storage-only-p :u16)))
    (signals error (ct:make-tensor-view storage '(4) :dtype :f32))))

(test make-tensor
  (let ((tensor (ct:make-tensor '(2 3 4) :dtype :f64)))
    (is (eq :f64 (ct:tensor-dtype tensor)))
    (is (= 3 (ct:tensor-rank tensor)))
    (is (= 24 (ct:tensor-size tensor)))
    (is (equalp #(12 4 1) (ct:tensor-strides tensor)))
    (is (= 0 (ct:tensor-offset tensor)))
    (is (= 24 (length (ct:tensor-storage tensor))))
    (is (every #'zerop (ct:tensor-storage tensor)))
    (is (ct:contiguous-p tensor))))

(test rank-zero-tensor
  (let ((tensor (ct:make-tensor '())))
    (is (= 0 (ct:tensor-rank tensor)))
    (is (= 1 (ct:tensor-size tensor)))
    (is (= 0 (ct:tensor-index tensor '())))
    (is (ct:contiguous-p tensor))
    (is (ct:inner-contiguous-p tensor))))

(test views-share-storage
  (let* ((storage (make-array 12 :element-type 'single-float))
         (a (ct:make-tensor-view storage '(3 4)))
         (b (ct:make-tensor-view storage '(4 3) :strides '(1 4))))
    (is (eq (ct:tensor-storage a) (ct:tensor-storage b)))
    (is (ct:contiguous-p a))
    (is (not (ct:contiguous-p b)))
    (is (not (ct:inner-contiguous-p b)))))

(test tensor-index
  (let ((tensor (ct:make-tensor-view (make-array 20 :element-type 'single-float) '(2 3)
                                     :strides '(-6 2) :offset 7)))
    (is (= 7 (ct:tensor-index tensor '(0 0))))
    (is (= 11 (ct:tensor-index tensor '(0 2))))
    (is (= 1 (ct:tensor-index tensor '(1 0))))
    (signals error (ct:tensor-index tensor '(2 0)))
    (signals error (ct:tensor-index tensor '(0 -1)))
    (signals error (ct:tensor-index tensor '(0)))))

(test view-validation
  (let ((storage (make-array 6 :element-type 'single-float)))
    (is (ct:make-tensor-view storage '(2 3)))
    (is (ct:make-tensor-view storage '(2 3) :strides '(-3 1) :offset 3))
    (is (ct:make-tensor-view storage '(2 3) :strides '(0 1)))
    (signals error (ct:make-tensor-view storage '(2 4)))
    (signals error (ct:make-tensor-view storage '(2 3) :offset 1))
    (signals error (ct:make-tensor-view storage '(2 3) :strides '(-3 1)))
    (signals error (ct:make-tensor-view storage '(2 3) :strides '(3)))
    (signals error (ct:make-tensor-view storage '(-1 3)))
    (signals error (ct:make-tensor-view storage '(2 3) :offset -1))
    (is (ct:make-tensor-view storage '(0 9) :offset 100))))

(test unit-axes-ignore-stride
  (let ((tensor (ct:make-tensor-view (make-array 3 :element-type 'single-float) '(1 3)
                                     :strides '(99 1))))
    (is (ct:contiguous-p tensor))))

(test print-object
  (is (string= "#<TENSOR :F32 (2 3)>"
               (princ-to-string (ct:make-tensor '(2 3))))))
