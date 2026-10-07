(require :asdf)
(asdf:load-system :cl-tensor)

(defpackage #:cl-tensor/extension-example
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor))
  (:export #:packed8 #:packed-storage #:make-packed #:decode #:run-example))

(in-package #:cl-tensor/extension-example)

(defclass packed8 (ct:dtype) ())

(defclass packed-storage ()
  ((codes :initarg :codes :reader codes)
   (scales :initarg :scales :reader scales)))

(defmethod ct:storage-length ((storage packed-storage)) (length (codes storage)))
(defmethod ct:storage-element-type ((storage packed-storage)) 'single-float)
(defmethod ct:storage-dtype ((storage packed-storage)) :example-packed8)

(defmethod ct:allocate-storage ((dtype packed8) shape)
  (let ((size (reduce #'* shape)))
    (unless (evenp size) (error "Packed storage requires pairs of elements"))
    (make-instance 'packed-storage
                   :codes (make-array size :element-type '(signed-byte 8) :initial-element 0)
                   :scales (make-array (/ size 2) :element-type 'single-float :initial-element 1f0))))

(defmethod ct:validate-storage-view ((dtype packed8) storage shape strides offset)
  (unless (and (typep storage 'packed-storage) (plusp (length shape))
               (evenp (aref shape (1- (length shape)))) (evenp offset)
               (equalp strides (ct:row-major-strides shape)))
    (error "Packed views require contiguous whole pairs along the last axis")))

(defmethod ct:copy-storage-supported-p ((dtype packed8) out)
  (typep (ct:tensor-storage out) 'packed-storage))

(defmethod ct:copy-storage! ((dtype packed8) out input)
  (unless (and (eq (ct:tensor-dtype out) (ct:tensor-dtype input))
               (equalp (ct:tensor-shape out) (ct:tensor-shape input)))
    (error "Packed copy shape and dtype must match"))
  (let ((source (ct:tensor-storage input)) (target (ct:tensor-storage out))
        (start (ct:tensor-offset input)) (end (ct:tensor-offset out))
        (size (ct:tensor-size out)))
    (replace (codes target) (codes source) :start1 end :end1 (+ end size)
                                          :start2 start :end2 (+ start size))
    (replace (scales target) (scales source) :start1 (/ end 2) :end1 (/ (+ end size) 2)
                                            :start2 (/ start 2) :end2 (/ (+ start size) 2)))
  out)

(defun decode (input)
  (let* ((out (ct:make-tensor (ct:tensor-shape input)))
         (storage (ct:tensor-storage input)) (start (ct:tensor-offset input)))
    (dotimes (i (ct:tensor-size input) out)
      (let ((index (+ start i)))
        (setf (ct:storage-ref (ct:tensor-storage out) i)
              (* (aref (codes storage) index) (aref (scales storage) (floor index 2))))))))

(defmethod ct:resolve-operation ((dtype packed8) operation inputs options)
  (let ((left (first inputs)) (right (second inputs)))
    (cond
      ((and (eq operation :negate) (eq (ct:tensor-dtype left) :example-packed8))
       (values :example-packed8
               (lambda (out inputs options)
                 (declare (ignore options))
                 (ct:copy-storage! dtype out (first inputs))
                 (let ((storage (ct:tensor-storage out)))
                   (dotimes (i (ct:tensor-size out))
                     (let ((value (- (aref (codes storage) i))))
                       (unless (typep value '(signed-byte 8))
                         (error "Negated value exceeds signed-byte storage"))
                       (setf (aref (codes storage) i) value)))))))
      ((and (eq operation :convert) (eq (ct:tensor-dtype left) :example-packed8)
            (eq (getf options :dtype) :example-packed8))
       (values :example-packed8
               (lambda (out inputs options)
                 (declare (ignore options))
                 (ct:copy-storage! dtype out (first inputs)))))
      ((and (eq operation :convert) (eq (ct:tensor-dtype left) :example-packed8)
            (eq (getf options :dtype) :f32))
       (values :f32 (lambda (out inputs options)
                      (declare (ignore options))
                      (ct:copy-storage! (ct:find-dtype :f32) out (decode (first inputs))))))
      ((and (member operation '(:dot :matmul))
            (eq (ct:tensor-dtype left) :example-packed8)
            (ct:tensorp right) (eq (ct:tensor-dtype right) :f32))
       (values :f32 (lambda (out inputs options)
                      (declare (ignore options))
                      (ct:matmul! out (decode (first inputs)) (second inputs))))))))

(ct:register-dtype
 (or (ignore-errors (ct:find-dtype :example-packed8))
     (make-instance 'packed8 :name :example-packed8 :element-type 'single-float
                            :block-size 2 :block-bytes 6 :storage-only t)))

(defun make-packed (shape values &optional (scale 1f0))
  (let* ((out (ct:make-tensor shape :dtype :example-packed8))
         (storage (ct:tensor-storage out)))
    (unless (= (length values) (ct:tensor-size out)) (error "Value count must match shape"))
    (replace (codes storage) values)
    (fill (scales storage) scale)
    out))

(defun run-example ()
  (let ((weights (make-packed '(2 2) '(1 2 3 4)))
        (activation (ct:from-data '(5 6))))
    (values (ct:matmul weights activation) (ct:dequantize weights))))
