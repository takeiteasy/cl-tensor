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
    (dolist (operation '(:negate :abs :sqrt :reciprocal :log :tanh :sigmoid :exp :sin :cos :silu :gelu))
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
                         :reciprocal :log :tanh :sigmoid :exp :sin :cos :silu :gelu :clamp :compare :select :sum :mean
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

(defvar *allocation-shapes* nil)
(defvar *record-allocations* nil)

(defmethod ct:allocate-storage :around ((dtype ct:dtype) shape)
  (when *record-allocations* (push (coerce shape 'list) *allocation-shapes*))
  (call-next-method))

(defclass failing-storage (wrapped-storage)
  ((reads :initform 0 :accessor reads)
   (fail-at :initarg :fail-at :reader fail-at)))

(defmethod ct:storage-ref ((storage failing-storage) index)
  (when (= (incf (reads storage)) (fail-at storage)) (error "Injected read failure"))
  (call-next-method))

(test take-custom-storage-and-hidden-aliases
  (let* ((input (wrapped '(1f0 2f0 3f0 4f0) '(2 2)))
         (out (ct:make-tensor-view (make-instance 'wrapped-storage
                                                :data (wrapped-data (ct:tensor-storage input)))
                                  '(2 2)))
         (indices (ct:from-data '(1 0) :dtype :s64)))
    (is (equalp '(3f0 4f0 1f0 2f0) (values-of (ct:take input indices))))
    (is (eq out (ct:take! out input indices)))
    (is (equalp '(3f0 4f0 1f0 2f0) (values-of out)))
    (signals ct:unsupported-storage-access
      (ct:take! (wrapped '(9f0 9f0 9f0 9f0) '(2 2) :writable nil) input indices)))
  (let* ((storage (make-instance 'failing-storage
                                 :data (make-array 4 :element-type 'single-float
                                                    :initial-contents '(1f0 2f0 3f0 4f0))
                                 :fail-at 3))
         (input (ct:make-tensor-view storage '(2 2)))
         (out (wrapped '(9f0 9f0 9f0 9f0) '(2 2))))
    (signals error (ct:take! out input (ct:from-data '(0 1) :dtype :s64)))
    (is (= 3 (reads storage)))
    (is (equalp '(9f0 9f0 9f0 9f0) (values-of out)))))

(test take-opaque-packed-storage
  (let* ((input (example:make-packed '(3 2) '(1 2 3 4 5 6) 2f0))
         (indices (ct:from-data '((2 0) (1 2)) :dtype :s64))
         (result (ct:take input indices))
         (out (example:make-packed '(2 2 2) '(0 0 0 0 0 0 0 0))))
    (is (eq :example-packed8 (ct:tensor-dtype result)))
    (is (equalp #(2 2 2) (ct:tensor-shape result)))
    (is (equalp '(10f0 12f0 2f0 4f0 6f0 8f0 10f0 12f0)
                (values-of (example:decode result))))
    (is (eq out (ct:take! out input indices)))
    (is (equalp (values-of (example:decode result)) (values-of (example:decode out))))
    (signals error (ct:take! out input (ct:from-data '((2 0) (1 3)) :dtype :s64)))
    (is (equalp (values-of (example:decode result)) (values-of (example:decode out))))
    (signals error (ct:take input (ct:from-data '(0 1) :dtype :s64) :axis 1)))
  (let* ((input (example:make-packed '(2 2) '(1 2 3 4)))
         (out (example:make-packed '(2 2) '(0 0 0 0))))
    (ct:take! out input (ct:from-data '(1 0) :dtype :s64))
    (is (equalp '(3f0 4f0 1f0 2f0) (values-of (example:decode out))))
    (ct:take! input input (ct:from-data '(1 0) :dtype :s64))
    (is (equalp '(3f0 4f0 1f0 2f0) (values-of (example:decode input))))))

(test bounded-elementwise-workspace-and-layouts
  (dolist (size '(0 1 4095 4096 4097 8193))
    (let* ((a (wrapped (make-list size :initial-element 3f0) (list size)))
           (out (ct:zeros (list size)))
           (*allocation-shapes* nil) (*record-allocations* t))
      (ct:add! out a a)
      (is (every (lambda (x) (= x 6f0)) (values-of out)))
      (is (= 3 (length *allocation-shapes*)))
      (is (= (if (> size 4096) 1 3) (count (list size) *allocation-shapes* :test #'equal)))
      (is (every (lambda (shape) (<= (first shape) 4096))
                 (remove (list size) *allocation-shapes* :test #'equal :count 1)))))
  (let* ((a (wrapped '(1f0 2f0 3f0 4f0 5f0 6f0) '(2 3)))
         (storage (ct:tensor-storage a))
         (reversed (ct:make-tensor-view storage '(2 3) :strides '(-3 -1) :offset 5))
         (out (ct:make-tensor-view storage '(2 3) :strides '(1 2))))
    (ct:add! out reversed (ct:from-data '((10 20 30))))
    (is (equalp '(16f0 25f0 34f0 13f0 22f0 31f0) (values-of out))))
  (let* ((a (wrapped '(1f0 2f0 3f0 4f0 5f0 6f0) '(6)))
         (input (ct:make-tensor-view (ct:tensor-storage a) '(5)))
         (out (ct:make-tensor-view (ct:tensor-storage a) '(5) :offset 1)))
    (ct:add! out input 10)
    (is (equalp '(1f0 11f0 12f0 13f0 14f0 15f0) (values-of a))))
  (let* ((a (wrapped '(2f0) nil))
         (out (wrapped '(0f0) nil)))
    (ct:multiply! out a 3)
    (is (= 6f0 (ct:tref out)))))

(test bounded-elementwise-families
  (let* ((a (wrapped '(1f0 2f0 3f0 4f0) '(2 2)))
         (ordinary (ct:from-data '((1 2) (3 4))))
         (mask (ct:compare :lt a 3)))
    (dolist (operation '(:add :subtract :multiply :divide :min :max))
      (let ((function (symbol-function (find-symbol (string operation) :cl-tensor))))
        (is (equalp (values-of (funcall function ordinary 2)) (values-of (funcall function a 2))))))
    (dolist (operation '(:negate :abs :sqrt :reciprocal :log :tanh :sigmoid :exp :sin :cos :silu :gelu))
      (let ((function (symbol-function (find-symbol (string operation) :cl-tensor))))
        (is (equalp (values-of (funcall function ordinary)) (values-of (funcall function a))))))
    (is (equalp '(1 1 0 0) (values-of mask)))
    (is (equalp '(1f0 2f0 9f0 9f0) (values-of (ct:select mask a 9))))
    (is (equalp '(2f0 2f0 3f0 3f0) (values-of (ct:clamp a 2 3))))))

(test inference-math-extension-out-forms
  (dolist (operation '(:exp :sin :cos :silu :gelu))
    (let* ((allocate (symbol-function (find-symbol (string operation) :cl-tensor)))
           (write (symbol-function (find-symbol (format nil "~A!" operation) :cl-tensor)))
           (input (wrapped '(-2f0 -1f0 0f0 1f0 2f0 3f0) '(2 3)))
           (out (wrapped '(0f0 0f0 0f0 0f0 0f0 0f0) '(2 3)))
           (expected (values-of (funcall allocate (ct:from-data '((-2 -1 0) (1 2 3)))))))
      (is (eq out (funcall write out input)))
      (is (equalp expected (values-of out)))
      (is (eq input (funcall write input input)))
      (is (equalp expected (values-of input))))
    (let* ((input (ct:from-data '(-2 -1 0 1 2 3) :dtype :example-hooked))
           (out (ct:zeros '(6)))
           (allocate (symbol-function (find-symbol (string operation) :cl-tensor)))
           (write (symbol-function (find-symbol (format nil "~A!" operation) :cl-tensor))))
      (is (eq out (funcall write out input)))
      (is (equalp (values-of (funcall allocate input)) (values-of out))))
    (let* ((storage (make-instance 'failing-storage
                                   :data (make-array 8193 :element-type 'single-float :initial-element 1f0)
                                   :fail-at 5000))
           (input (ct:make-tensor-view storage '(8193)))
           (out (ct:full '(8193) 7))
           (write (symbol-function (find-symbol (format nil "~A!" operation) :cl-tensor))))
      (signals error (funcall write out input))
      (is (every (lambda (x) (= 7 x)) (values-of out))))))

(test bounded-fallback-validation-and-failure-atomicity
  (let* ((storage (make-instance 'failing-storage :data (make-array 8193 :element-type 'single-float
                                                                 :initial-element 2f0) :fail-at 5000))
         (a (ct:make-tensor-view storage '(8193)))
         (out (ct:full '(8193) 7)))
    (signals error (ct:add! (ct:zeros '(2)) a a))
    (signals error (ct:add! (ct:zeros '(8193) :dtype :f64) a a))
    (signals error (ct:add! out a (ct:zeros '(8193) :dtype :f64)))
    (is (= 0 (reads storage)))
    (signals error (ct:add! out a a))
    (is (every (lambda (x) (= x 7f0)) (values-of out))))
  ;; ECL's compiled calls bypass SYMBOL-FUNCTION replacement.
  #+sbcl
  (let* ((a (wrapped (make-list 8193 :initial-element 2f0) '(8193)))
         (out (ct:full '(8193) 7))
         (original (symbol-function 'ct::run-nd)) (calls 0))
    (unwind-protect
         (progn
           (setf (symbol-function 'ct::run-nd)
                 (lambda (&rest arguments)
                   (when (= (incf calls) 2) (error "Injected kernel failure"))
                   (apply original arguments)))
           (signals error (ct:add! out a a)))
      (setf (symbol-function 'ct::run-nd) original))
    (is (= 2 calls))
    (is (every (lambda (x) (= x 7f0)) (values-of out)))))

(test direct-opaque-execution-and-decline
  (let* ((a (example:make-packed '(4) '(1 2 3 4)))
         (out (ct:zeros '(4) :dtype :example-packed8))
         (*record-allocations* t) (*allocation-shapes* nil))
    (is (eq out (ct:negate! out a)))
    (is (null *allocation-shapes*))
    (is (equalp '(-1f0 -2f0 -3f0 -4f0) (values-of (ct:dequantize out))))
    (setf *allocation-shapes* nil)
    (is (eq a (ct:negate! a a)))
    (is (null *allocation-shapes*)))
  (let* ((a (example:make-packed '(6) '(1 2 3 4 5 6)))
         (input (ct:slice a :selectors '((0 4))))
         (out (ct:slice a :selectors '((2 6))))
         (*record-allocations* t) (*allocation-shapes* nil))
    (ct:negate! out input)
    (is (equal '((4)) *allocation-shapes*))
    (is (equalp '(1f0 2f0 -1f0 -2f0 -3f0 -4f0) (values-of (ct:dequantize a)))))
  (let ((input (example:make-packed '(4) '(1 2 3 -128)))
        (out (example:make-packed '(4) '(7 7 7 7))))
    (signals error (ct:negate! out input))
    (is (equalp '(7f0 7f0 7f0 7f0) (values-of (ct:dequantize out))))))

(defvar *direct-mode* nil)
(defvar *direct-selections* 0)
(defvar *direct-executions* 0)
(defclass direct-test-dtype (ct:dtype) ())

(defmethod ct:resolve-operation ((dtype direct-test-dtype) operation inputs options)
  (declare (ignore dtype operation inputs options))
  (values :f32
          (lambda (out inputs options)
            (declare (ignore inputs options))
            (incf *direct-executions*)
            (setf (ct:tref out 0) 99f0)
            (error "Injected staged executor failure"))
          (if (eq *direct-mode* :malformed-selector) 42
              (lambda (out inputs options)
                (declare (ignore out inputs options))
                (incf *direct-selections*)
                (case *direct-mode* (:malformed-executor 42) (t nil))))))

(ct:register-dtype
 (or (ignore-errors (ct:find-dtype :example-direct-test))
     (make-instance 'direct-test-dtype :name :example-direct-test :element-type 'single-float
                                      :block-bytes 4 :zero 0f0)))

(test direct-selector-validation-and-staged-failure
  (let ((a (ct:zeros '(2) :dtype :example-direct-test)) (out (ct:full '(2) 7))
        (*direct-selections* 0) (*direct-executions* 0))
    (signals error (ct:add! (ct:zeros '(3)) a a))
    (signals error (ct:add! (ct:zeros '(2) :dtype :f64) a a))
    (signals error (ct:add! (ct:make-tensor-view (ct:tensor-storage out) '(2) :strides '(0)) a a))
    (signals error (ct:add! (wrapped '(0f0 0f0) '(2) :writable nil) a a))
    (is (= 0 *direct-selections*))
    (is (= 0 *direct-executions*))
    (dolist (*direct-mode* '(:malformed-selector :malformed-executor))
      (signals error (ct:add! out a a))
      (is (equalp '(7f0 7f0) (values-of out)))
      (is (= 0 *direct-executions*)))
    (signals error (ct:add! out a a))
    (is (= 1 *direct-executions*))
    (is (equalp '(7f0 7f0) (values-of out)))))

(defclass typed-wrapped-storage (wrapped-storage) ())
(defmethod ct:storage-element-type ((storage typed-wrapped-storage))
  (array-element-type (wrapped-data storage)))

(test bounded-fallback-dtypes
  (dolist (dtype '(:f32 :f64 :s8 :u8 :s16 :u16 :s32 :u32 :s64 :u64 :c32 :c64))
    (let* ((ordinary (ct:full '(4097) 2 :dtype dtype))
           (a (ct:make-tensor-view (make-instance 'typed-wrapped-storage :data (ct:tensor-storage ordinary))
                                   '(4097) :dtype dtype))
           (out (ct:zeros '(4097) :dtype dtype)))
      (ct:add! out a a)
      (is (every (lambda (x) (= x 4)) (values-of out)))
      (is (equalp (values-of (ct:abs ordinary)) (values-of (ct:abs a))))))
  (let* ((mask (ct:make-tensor-view
                (make-instance 'typed-wrapped-storage
                               :data (ct:tensor-storage (ct:full '(4097) 1 :dtype :u8)))
                '(4097) :dtype :u8))
         (out (wrapped (make-list 4097 :initial-element 0f0) '(4097))))
    (ct:select! out mask 3 9)
    (is (every (lambda (x) (= x 3f0)) (values-of out)))))

(test opaque-storage-geometry-and-hidden-aliases
  (signals error
    (ct:make-tensor-view
     (make-instance 'example:packed-storage
                    :codes (make-array 4 :element-type '(signed-byte 8) :initial-element 1)
                    :scales (make-array 1 :element-type 'single-float :initial-element 1f0)) '(4)))
  (let* ((base (example:make-packed '(6) '(1 2 3 4 5 6)))
         (storage (ct:tensor-storage base))
         (wrapper (make-instance 'example:packed-storage :codes (example::codes storage)
                                                         :scales (example::scales storage)))
         (input (ct:make-tensor-view storage '(4)))
         (out (ct:make-tensor-view wrapper '(4) :offset 2)))
    (is (null (example::select-direct-negate out (list input) nil)))
    (ct:negate! out input)
    (is (equalp '(1f0 2f0 -1f0 -2f0 -3f0 -4f0) (values-of (ct:dequantize base))))))

(test operation-table-direct-layouts
  (let* ((descriptor
           (or (ignore-errors (ct:find-dtype :example-direct-table))
               (ct:register-dtype
                (make-instance 'ct:dtype :name :example-direct-table :element-type 'single-float
                               :zero 0f0 :block-bytes 4
                               :operations
                               (list :add
                                     (lambda (dtype operation inputs options)
                                       (declare (ignore dtype operation inputs options))
                                       (let ((executor (lambda (out inputs options)
                                                         (declare (ignore inputs options))
                                                         (dotimes (i (ct:tensor-size out))
                                                           (setf (ct:tref out i) 7f0)))))
                                         (values :f32 executor
                                                 (lambda (out inputs options)
                                                   (declare (ignore inputs options))
                                                   (when (typep (ct:tensor-storage out) '(simple-array single-float (*)))
                                                     executor))))))))))
         (a (ct:zeros '(2) :dtype (ct:dtype-name descriptor)))
         (base (ct:full '(6) 9)))
    (dolist (layout '(((-1) 3) ((2) 1)))
      (destructuring-bind (strides offset) layout
        (let ((out (ct:make-tensor-view (ct:tensor-storage base) '(2) :strides strides :offset offset))
              (*allocation-shapes* nil) (*record-allocations* t))
          (is (eq out (ct:add! out a (ct:ones '(2)))))
          (is (equalp '(7f0 7f0) (values-of out)))
          (is (equal '((2)) *allocation-shapes*)))))))
