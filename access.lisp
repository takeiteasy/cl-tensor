(in-package #:cl-tensor)

(defun computational-dtype (dtype)
  (when (dtype-storage-only-p dtype)
    (error "Dtype ~S is storage only; convert with ASTYPE before computing" dtype))
  dtype)

(defun coerce-value (value dtype)
  (let ((type (dtype-element-type dtype)))
    (cond ((member dtype '(:f32 :f64))
           (unless (realp value) (error 'type-error :datum value :expected-type 'real))
           (coerce value type))
          ((member dtype '(:c32 :c64))
           (unless (numberp value) (error 'type-error :datum value :expected-type 'number))
           (let ((real-type (if (eq dtype :c32) 'single-float 'double-float)))
             (complex (coerce (realpart value) real-type)
                      (coerce (imagpart value) real-type))))
          (t
           (unless (typep value type)
             (error 'type-error :datum value :expected-type type))
           value))))

(defmacro define-foreign-access ()
  (let ((types '((:f32 :float) (:f64 :double)
                 (:s8 :int8) (:u8 :uint8) (:s16 :int16) (:u16 :uint16)
                 (:s32 :int32) (:u32 :uint32) (:s64 :int64) (:u64 :uint64)
                 (:c32 :float) (:c64 :double))))
    `(progn
       (defun foreign-ref (storage index)
         (let ((pointer (trivial-simd:vector-view-pointer storage)))
           (ecase (trivial-simd:vector-view-type storage)
             ,@(loop for (dtype foreign) in types
                     collect `(,dtype
                               ,(if (member dtype '(:c32 :c64))
                                    `(complex (cffi:mem-aref pointer ,foreign (* 2 index))
                                              (cffi:mem-aref pointer ,foreign (1+ (* 2 index))))
                                    `(cffi:mem-aref pointer ,foreign index)))))))
       (defun (setf foreign-ref) (value storage index)
         (let ((pointer (trivial-simd:vector-view-pointer storage)))
           (ecase (trivial-simd:vector-view-type storage)
             ,@(loop for (dtype foreign) in types
                     collect `(,dtype
                               ,(if (member dtype '(:c32 :c64))
                                    `(setf (cffi:mem-aref pointer ,foreign (* 2 index)) (realpart value)
                                           (cffi:mem-aref pointer ,foreign (1+ (* 2 index))) (imagpart value))
                                    `(setf (cffi:mem-aref pointer ,foreign index) value))))))
         value))))

(define-foreign-access)

(defun storage-ref (storage index)
  (if (trivial-simd:vector-view-p storage)
      (foreign-ref storage index)
      (aref storage index)))

(defun (setf storage-ref) (value storage index)
  (if (trivial-simd:vector-view-p storage)
      (setf (foreign-ref storage index) value)
      (setf (aref storage index) value)))

(defun tref (tensor &rest indices)
  (storage-ref (tensor-storage tensor) (tensor-index tensor indices)))

(defun (setf tref) (value tensor &rest indices)
  (let ((index (tensor-index tensor indices)))
    (setf (storage-ref (tensor-storage tensor) index)
          (coerce-value value (tensor-dtype tensor))))
  value)

(defun call-with-offsets (shape layouts offsets function)
  (labels ((walk (axis)
             (if (= axis (length shape))
                 (apply function offsets)
                 (let ((count (aref shape axis)))
                   (dotimes (i count)
                     (walk (1+ axis))
                     (loop for tail on offsets for strides in layouts
                           do (incf (car tail) (aref strides axis))))
                   (loop for tail on offsets for strides in layouts
                         do (decf (car tail) (* count (aref strides axis))))))))
    (walk 0)))

(defun copy-tensor (tensor)
  (let* ((result (make-tensor (tensor-shape tensor) :dtype (tensor-dtype tensor)))
         (target (tensor-storage result))
         (position 0))
    (call-with-offsets (tensor-shape tensor) (list (tensor-strides tensor))
                       (list (tensor-offset tensor))
                       (lambda (index)
                         (setf (aref target position) (storage-ref (tensor-storage tensor) index))
                         (incf position)))
    result))
