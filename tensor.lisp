(in-package #:cl-tensor)

(deftype index-vector () '(simple-array fixnum (*)))

(defstruct (tensor (:constructor %make-tensor (storage dtype shape strides offset))
                   (:copier nil)
                   (:predicate tensorp))
  (storage nil :read-only t)
  (dtype nil :type keyword :read-only t)
  (shape nil :type index-vector :read-only t)
  (strides nil :type index-vector :read-only t)
  (offset 0 :type fixnum :read-only t))

(defmethod print-object ((tensor tensor) stream)
  (print-unreadable-object (tensor stream :type t)
    (format stream "~S ~S" (tensor-dtype tensor) (coerce (tensor-shape tensor) 'list))))

(defun index-vector (sequence)
  (make-array (length sequence) :element-type 'fixnum :initial-contents sequence))

(defun tensor-rank (tensor)
  (length (tensor-shape tensor)))

(defun tensor-size (tensor)
  (reduce #'* (tensor-shape tensor)))

(defun row-major-strides (shape)
  (let* ((rank (length shape))
         (strides (make-array rank :element-type 'fixnum))
         (stride 1))
    (loop for axis from (1- rank) downto 0
          do (setf (aref strides axis) stride)
             (setf stride (* stride (elt shape axis))))
    strides))

(defun check-shape (shape)
  (unless (every (lambda (dimension) (typep dimension '(integer 0 #.most-positive-fixnum)))
                 shape)
    (error "Shape must be non-negative integers: ~S" shape)))

(defun reachable-range (shape strides offset)
  "Lowest and highest storage index reached by the view, assuming it is non-empty."
  (let ((low offset) (high offset))
    (dotimes (axis (length shape))
      (let ((extent (* (aref strides axis) (1- (aref shape axis)))))
        (if (minusp extent)
            (incf low extent)
            (incf high extent))))
    (values low high)))

(defun make-tensor-view (storage shape &key (dtype (storage-dtype storage)) strides (offset 0))
  (check-shape shape)
  (let* ((shape (index-vector shape))
         (strides (if strides (index-vector strides) (row-major-strides shape))))
    (unless (= (length strides) (length shape))
      (error "Strides ~S do not match the rank of shape ~S" strides shape))
    (unless (typep offset '(integer 0 #.most-positive-fixnum))
      (error "Offset must be a non-negative integer: ~S" offset))
    (unless (zerop (reduce #'* shape))
      (multiple-value-bind (low high) (reachable-range shape strides offset)
        (unless (and (<= 0 low) (< high (storage-length storage)))
          (error "View reaches storage indices ~D to ~D, outside storage of length ~D"
                 low high (storage-length storage)))))
    (validate-storage-view (find-dtype dtype) storage shape strides offset)
    (%make-tensor storage dtype shape strides offset)))

(defun make-tensor (shape &key (dtype :f32))
  (check-shape shape)
  (make-tensor-view (allocate-storage (find-dtype dtype) shape) shape :dtype dtype))

(defun tensor-index (tensor indices)
  "Storage index of the element at INDICES, a list with one entry per axis."
  (let ((shape (tensor-shape tensor))
        (strides (tensor-strides tensor)))
    (unless (= (length indices) (length shape))
      (error "Expected ~D indices, got ~D" (length shape) (length indices)))
    (let ((index (tensor-offset tensor)))
      (loop for axis from 0
            for i in indices
            do (unless (and (integerp i) (< -1 i (aref shape axis)))
                 (error "Index ~S out of bounds for axis ~D of size ~D" i axis (aref shape axis)))
               (incf index (* i (aref strides axis))))
      index)))

(defun contiguous-p (tensor)
  "True when the elements occupy consecutive storage in row-major order."
  (let ((shape (tensor-shape tensor))
        (strides (tensor-strides tensor))
        (expected 1))
    (or (zerop (tensor-size tensor))
        (loop for axis from (1- (length shape)) downto 0
              always (or (= (aref shape axis) 1)
                         (= (aref strides axis) expected))
              do (setf expected (* expected (aref shape axis)))))))

(defun inner-contiguous-p (tensor)
  "True when the innermost axis has unit stride, the kernel fast path."
  (let ((shape (tensor-shape tensor)))
    (or (zerop (length shape))
        (let ((axis (1- (length shape))))
          (or (<= (aref shape axis) 1)
              (= (aref (tensor-strides tensor) axis) 1))))))
