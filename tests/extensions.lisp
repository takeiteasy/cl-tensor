(defpackage #:cl-tensor/extension-tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:ct #:cl-tensor) (#:example #:cl-tensor/extension-example)))

(in-package #:cl-tensor/extension-tests)
(in-suite :cl-tensor)

(defclass wrapped-storage ()
  ((data :initarg :data :reader wrapped-data)
   (writable :initarg :writable :initform t :reader writable)))

(defmethod ct:storage-length ((storage wrapped-storage)) (length (wrapped-data storage)))
(defmethod ct:storage-element-type ((storage wrapped-storage)) 'single-float)
(defmethod ct:storage-readable-p ((storage wrapped-storage)) t)
(defmethod ct:storage-writable-p ((storage wrapped-storage)) (writable storage))
(defmethod ct:storage-ref ((storage wrapped-storage) index) (aref (wrapped-data storage) index))
(defmethod (setf ct:storage-ref) (value (storage wrapped-storage) index)
  (unless (writable storage)
    (error 'ct:unsupported-storage-access :storage storage :access :write))
  (setf (aref (wrapped-data storage) index) value))

(defun wrapped (values shape &key (writable t))
  (ct:make-tensor-view (make-instance 'wrapped-storage
                                    :data (make-array (length values) :element-type 'single-float
                                                                   :initial-contents values)
                                    :writable writable)
                      shape))

(defun values-of (tensor)
  (let ((flat (ct:reshape tensor (list (ct:tensor-size tensor)))))
    (loop for i below (ct:tensor-size flat) collect (ct:tref flat i))))

(defclass hooked-dtype (ct:dtype) ())
(defmethod ct:coerce-scalar ((dtype hooked-dtype) value)
  (coerce value 'single-float))

(defvar *selections* nil)
(defvar *executions* 0)

(defun ordinary (input)
  (if (and (ct:tensorp input) (typep (ct:find-dtype (ct:tensor-dtype input)) 'hooked-dtype))
      (ct:make-tensor-view (ct:tensor-storage input) (ct:tensor-shape input)
                          :strides (ct:tensor-strides input) :offset (ct:tensor-offset input) :dtype :f32)
      input))

(defmethod ct:resolve-operation ((dtype hooked-dtype) operation inputs options)
  (push operation *selections*)
  (let ((name (case operation (:compare :u8) ((:argmin :argmax) :s64)
                               ((:convert :dequantize) (getf options :dtype)) (t :f32))))
    (values name
            (lambda (out inputs options)
              (incf *executions*)
              (let* ((inputs (mapcar #'ordinary inputs))
                     (result
                       (case operation
                         ((:convert :dequantize) (ct:astype (first inputs) (getf options :dtype)))
                         ((:dot :matmul) (ct:matmul (first inputs) (second inputs)))
                         (:compare (ct:compare (getf options :operator) (first inputs) (second inputs)))
                         ((:sum :mean :prod :minimum :maximum :argmin :argmax)
                          (apply (symbol-function (find-symbol (string operation) :cl-tensor))
                                 (first inputs) :axis (coerce (getf options :axis) 'vector)
                                 :keepdims (getf options :keepdims)
                                 (when (member operation '(:sum :mean))
                                   (list :accumulate (getf options :accumulate)))))
                         (:softmax (ct:softmax (first inputs) :axis (coerce (getf options :axis) 'vector)))
                         (:rmsnorm (ct:rmsnorm (first inputs) :weights (second inputs)
                                              :epsilon (getf options :epsilon)
                                              :axis (coerce (getf options :axis) 'vector)))
                         (t (apply (symbol-function (find-symbol (string operation) :cl-tensor)) inputs)))))
                (ct:copy-storage! (ct:find-dtype name) out result))))))

(ct:register-dtype
 (or (ignore-errors (ct:find-dtype :example-hooked))
     (make-instance 'hooked-dtype :name :example-hooked :element-type 'single-float
                                 :block-bytes 4 :zero 0f0)))

(test extension-registry
  (let ((descriptor (ct:find-dtype :example-packed8)))
    (is (eq descriptor (ct:register-dtype descriptor)))
    (is (= 2 (ct:dtype-block-size :example-packed8)))
    (is (= 6 (ct:dtype-block-bytes :example-packed8)))
    (is (= 3 (ct:dtype-bytes :example-packed8)))
    (signals error (ct:register-dtype (make-instance 'ct:dtype :name :example-packed8
                                                            :element-type 'single-float :block-bytes 4))))
  (signals error (ct:register-dtype (make-instance 'ct:dtype :name :invalid-block
                                                          :element-type 'single-float :block-size 0
                                                          :block-bytes 4)))
  (is (eq :f32 (ct:storage-dtype (make-array 2 :element-type 'single-float))))
  (is (eq :u16 (ct:storage-dtype (make-array 2 :element-type '(unsigned-byte 16)))))
  (signals error (ct:find-dtype :unregistered)))

(test opaque-packed-extension
  (let* ((weights (example:make-packed '(2 2) '(1 2 3 4) 2f0))
         (x (ct:from-data '(5 6)))
         (decoded (ct:dequantize weights)))
    (is (equalp '(34f0 78f0) (values-of (ct:matmul weights x))))
    (is (equalp '(2f0 4f0 6f0 8f0) (values-of decoded)))
    (is (equalp (values-of decoded) (values-of (ct:astype weights :f32))))
    (is (not (eq (ct:tensor-storage decoded) (ct:tensor-storage weights))))
    (is (equalp (values-of decoded) (values-of (ct:dequantize (ct:copy-tensor weights)))))
    (is (not (ct:storage-readable-p (ct:tensor-storage weights))))
    (is (not (ct:storage-writable-p (ct:tensor-storage weights))))
    (signals ct:unsupported-storage-access (ct:tref weights 0 0))
    (signals ct:unsupported-storage-access (setf (ct:tref weights 0 0) 1))
    (signals ct:unsupported-operation (ct:add weights weights))
    (signals error (ct:transpose weights))
    (signals error (ct:slice weights :selectors '(:all (0 1))))
    (signals error (ct:make-tensor-view (ct:tensor-storage weights) '(2 2) :offset 2))
    (signals error (ct:make-tensor-view (ct:tensor-storage weights) '(2) :offset 1))
    (signals error (ct:from-data '((1 2)) :dtype :example-packed8))
    (is (equalp '(0f0 0f0) (values-of (ct:dequantize (ct:zeros '(2) :dtype :example-packed8)))))
    (let ((out (ct:make-tensor-view (ct:tensor-storage x) '(2) :strides '(-1) :offset 1)))
      (is (eq out (ct:matmul! out weights x)))
      (is (equalp '(34f0 78f0) (values-of out))))
    (let ((row (ct:slice weights :selectors '((1 2)))))
      (is (equalp '(6f0 8f0) (values-of (ct:dequantize row)))))))

(test extension-hooks-cover-operation-families
  (let* ((a (ct:from-data '((1 2) (3 4)) :dtype :example-hooked))
         (v (ct:from-data '(1 2) :dtype :example-hooked))
         (*selections* nil) (*executions* 0))
    (dolist (operation '(:add :subtract :multiply :divide :min :max))
      (let ((function (symbol-function (find-symbol (string operation) :cl-tensor))))
        (is (equalp (values-of (funcall function (ordinary a) (ordinary a)))
                    (values-of (funcall function a a))))))
    (dolist (operation '(:negate :abs :sqrt :reciprocal :log :tanh :sigmoid))
      (let ((function (symbol-function (find-symbol (string operation) :cl-tensor))))
        (is (equalp (values-of (funcall function (ordinary a))) (values-of (funcall function a))))))
    (ct:clamp a 1 3)
    (ct:compare :lt a 3)
    (ct:select (ct:compare :lt a 3) a a)
    (dolist (operation '(:sum :mean :prod :minimum :maximum :argmin :argmax))
      (let ((function (symbol-function (find-symbol (string operation) :cl-tensor))))
        (is (equalp (values-of (funcall function (ordinary a) :axis 0))
                    (values-of (funcall function a :axis 0))))))
    (ct:softmax a)
    (ct:rmsnorm a :weights v)
    (ct:matmul a a)
    (is (= 5f0 (ct:tref (ct:matmul v v))))
    (ct:astype a :f32)
    (ct:dequantize a)
    (dolist (operation '(:add :subtract :multiply :divide :min :max :negate :abs :sqrt
                         :reciprocal :log :tanh :sigmoid :clamp :compare :select :sum :mean
                         :prod :minimum :maximum :argmin :argmax :softmax :rmsnorm :matmul
                         :dot :convert :dequantize))
      (is (not (null (member operation *selections*)))))))

(test accessible-storage-fallback
  (let* ((a (wrapped '(1f0 2f0 3f0 4f0) '(2 2)))
         (b (ct:from-data '((1 2) (3 4))))
         (out (wrapped '(0f0 0f0 0f0 0f0) '(2 2))))
    (is (equalp '(2f0 4f0 6f0 8f0) (values-of (ct:add a a))))
    (is (equalp '(4f0 6f0) (values-of (ct:sum a :axis 0))))
    (is (equalp (values-of (ct:softmax b)) (values-of (ct:softmax a))))
    (is (equalp (values-of (ct:rmsnorm b)) (values-of (ct:rmsnorm a))))
    (is (eq out (ct:matmul! out a a)))
    (is (equalp '(7f0 10f0 15f0 22f0) (values-of out)))
    (is (equalp '(1f0 2f0 3f0 4f0) (values-of (ct:astype a :f32))))
    (is (eq a (ct:add! a a a)))
    (is (equalp '(2f0 4f0 6f0 8f0) (values-of a)))
    (signals ct:unsupported-storage-access
      (ct:add! (wrapped '(0f0 0f0 0f0 0f0) '(2 2) :writable nil) a a))))

(test hook-output-validation-and-staging
  (let* ((a (ct:from-data '(1 2) :dtype :example-hooked))
         (storage (ct:tensor-storage a))
         (out (ct:make-tensor-view storage '(2) :dtype :f32 :strides '(-1) :offset 1))
         (*executions* 0))
    (ct:add! out a a)
    (is (equalp '(2f0 4f0) (values-of out)))
    (let ((before *executions*))
      (signals error (ct:add! (ct:zeros '(3)) a a))
      (signals error (ct:add! (ct:zeros '(2) :dtype :f64) a a))
      (signals error (ct:add! (ct:make-tensor-view storage '(2) :strides '(0)) a a))
      (signals ct:unsupported-storage-access
        (ct:add! (wrapped '(0f0 0f0) '(2) :writable nil) a a))
      (is (= before *executions*)))))

(test operation-table-and-declined-hooks
  (let* ((descriptor
           (or (ignore-errors (ct:find-dtype :example-table))
               (ct:register-dtype
                (make-instance 'ct:dtype :name :example-table :element-type 'single-float
                               :zero 0f0 :block-bytes 4
                               :operations
                               (list :add
                                     (lambda (dtype operation inputs options)
                                       (declare (ignore dtype operation inputs options))
                                       (values :f32
                                               (lambda (out inputs options)
                                                 (declare (ignore inputs options))
                                                 (dotimes (i (ct:tensor-size out))
                                                   (setf (ct:storage-ref (ct:tensor-storage out) i) 7f0))))))))))
         (a (ct:make-tensor '(2) :dtype (ct:dtype-name descriptor))))
    (is (equalp '(7f0 7f0) (values-of (ct:add (ct:ones '(2)) a))))
    (signals ct:unsupported-operation (ct:multiply a a)))
  (let ((descriptor (or (ignore-errors (ct:find-dtype :example-fraction))
                        (ct:register-dtype (make-instance 'ct:dtype :name :example-fraction
                                                         :element-type 'single-float :block-size 32
                                                         :block-bytes 34)))))
    (is (eq :example-fraction (ct:dtype-name descriptor)))
    (is (= 17/16 (ct:dtype-bytes :example-fraction)))))

(test packed-shape-materialization-and-dot
  (let* ((v (example:make-packed '(2) '(1 2)))
         (joined (ct:concatenate (list v v))))
    (is (= 11f0 (ct:tref (ct:matmul v (ct:from-data '(3 4))))))
    (is (equalp '(1f0 2f0 1f0 2f0) (values-of (ct:dequantize joined))))
    (is (equalp '(1f0 2f0) (values-of (ct:dequantize (ct:reshape v '(1 2))))))))

(test extension-validation-before-execution
  (let ((a (ct:ones '(2 2) :dtype :example-hooked))
        (*executions* 0))
    (signals error (ct:matmul a (ct:ones '(3 2))))
    (signals error (ct:add a (ct:ones '(3))))
    (signals error (ct:sum a :axis '(0 0)))
    (signals error (ct:sum a :axis 3))
    (signals error (ct:softmax a :axis 3))
    (signals error (ct:rmsnorm a :epsilon -1))
    (signals error (ct:rmsnorm a :weights (ct:ones '(3))))
    (signals error (ct:compare :unknown a a))
    (signals error (ct:select a a a))
    (signals error (ct:clamp a 3 1))
    (signals error (ct:astype a :f32 :rounding :unknown))
    (signals error (ct:mean (ct:zeros '(0) :dtype :example-hooked)))
    (is (= 0 *executions*))))

(test opaque-hook-destinations-and-failure-atomicity
  (let ((a (example:make-packed '(2) '(1 2)))
        (out (ct:zeros '(2) :dtype :example-packed8)))
    (is (eq out (ct:negate! out a)))
    (is (equalp '(-1f0 -2f0) (values-of (ct:dequantize out))))
    (is (eq a (ct:negate! a a)))
    (is (equalp '(-1f0 -2f0) (values-of (ct:dequantize a))))
    (is (equalp (values-of (ct:dequantize a))
                (values-of (ct:dequantize (ct:astype a :example-packed8)))))
    (let ((invalid (example:make-packed '(2) '(-128 1))))
      (signals error (ct:negate! out invalid))
      (is (equalp '(-1f0 -2f0) (values-of (ct:dequantize out)))))))

(defclass wrapped-dtype (hooked-dtype) ())

(defmethod ct:allocate-storage ((dtype wrapped-dtype) shape)
  (make-instance 'wrapped-storage
                 :data (make-array (reduce #'* shape) :element-type 'single-float :initial-element 0f0)))

(ct:register-dtype
 (or (ignore-errors (ct:find-dtype :example-wrapped))
     (make-instance 'wrapped-dtype :name :example-wrapped :element-type 'single-float
                                  :block-bytes 4 :zero 0f0)))

(test custom-storage-constructors-and-copy
  (let ((a (ct:from-data '((1 2) (3 4)) :dtype :example-wrapped)))
    (is (typep (ct:tensor-storage a) 'wrapped-storage))
    (is (equalp '(1f0 2f0 3f0 4f0) (values-of a)))
    (setf (ct:tref a 0 0) 9)
    (is (= 9f0 (ct:tref a 0 0)))
    (is (equalp (values-of a) (values-of (ct:copy-tensor a))))
    (is (not (eq (ct:tensor-storage a) (ct:tensor-storage (ct:copy-tensor a)))))
    (is (equalp '(9f0 2f0 3f0 4f0 9f0 2f0 3f0 4f0)
                (values-of (ct:concatenate (list a a)))))
    (is (equalp '(1f0 1f0 1f0) (values-of (ct:ones '(3) :dtype :example-wrapped))))
    (is (equalp '(0f0 1f0 2f0) (values-of (ct:arange 3 :dtype :example-wrapped))))
    (is (equalp '(1f0 0f0 0f0 1f0) (values-of (ct:eye 2 :dtype :example-wrapped))))))

(test conversion-destination-selection-and-selector-errors
  (let* ((descriptor
           (or (ignore-errors (ct:find-dtype :example-target))
               (ct:register-dtype
                (make-instance 'ct:dtype :name :example-target :element-type 'single-float
                               :zero 0f0 :block-bytes 4
                               :operations
                               (list :convert
                                     (lambda (dtype operation inputs options)
                                       (declare (ignore operation inputs options))
                                       (values (ct:dtype-name dtype)
                                               (lambda (out inputs options)
                                                 (declare (ignore options))
                                                 (loop for value in (values-of (first inputs))
                                                       for i from 0 do
                                                         (setf (ct:storage-ref (ct:tensor-storage out) i)
                                                               value))))))))))
         (result (ct:astype (ct:from-data '(1 2)) (ct:dtype-name descriptor))))
    (is (eq :example-target (ct:tensor-dtype result)))
    (is (equalp '(1f0 2f0) (values-of result))))
  (let* ((descriptor
           (or (ignore-errors (ct:find-dtype :example-invalid-selector))
               (ct:register-dtype
                (make-instance 'ct:dtype :name :example-invalid-selector :element-type 'single-float
                               :zero 0f0 :block-bytes 4
                               :operations (list :add (lambda (&rest arguments)
                                                        (declare (ignore arguments))
                                                        (values :f32 nil))
                                                 :convert (lambda (&rest arguments)
                                                            (declare (ignore arguments))
                                                            (values :f64 (lambda (&rest arguments)
                                                                           (declare (ignore arguments))
                                                                           (error "Must not execute")))))))))
         (a (ct:make-tensor '(2) :dtype (ct:dtype-name descriptor)))
         (out (ct:ones '(2))))
    (signals error (ct:add! out a a))
    (is (equalp '(1f0 1f0) (values-of out)))
    (signals error (ct:astype a :f32))))
