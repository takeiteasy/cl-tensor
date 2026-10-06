(in-package #:cl-tensor)

(defstruct (dtype-info (:conc-name dtype-info-))
  (name nil :read-only t)
  (element-type nil :read-only t)
  (bytes 0 :read-only t)
  (zero nil :read-only t)
  (storage-only nil :read-only t))

(defparameter *dtypes*
  (flet ((info (name element-type bytes zero &optional storage-only)
           (make-dtype-info :name name :element-type element-type :bytes bytes
                            :zero zero :storage-only storage-only)))
    (list (info :f32 'single-float 4 0f0)
          (info :f64 'double-float 8 0d0)
          (info :c32 '(complex single-float) 8 #C(0f0 0f0))
          (info :c64 '(complex double-float) 16 #C(0d0 0d0))
          (info :s8 '(signed-byte 8) 1 0)
          (info :u8 '(unsigned-byte 8) 1 0)
          (info :s16 '(signed-byte 16) 2 0)
          (info :u16 '(unsigned-byte 16) 2 0)
          (info :s32 '(signed-byte 32) 4 0)
          (info :u32 '(unsigned-byte 32) 4 0)
          (info :s64 '(signed-byte 64) 8 0)
          (info :u64 '(unsigned-byte 64) 8 0)
          (info :f16 '(unsigned-byte 16) 2 0 t)
          (info :bf16 '(unsigned-byte 16) 2 0 t))))

(defun find-dtype (dtype)
  (or (find dtype *dtypes* :key #'dtype-info-name)
      (error "Unknown dtype ~S" dtype)))

(defun dtype-element-type (dtype)
  (dtype-info-element-type (find-dtype dtype)))

(defun dtype-bytes (dtype)
  (dtype-info-bytes (find-dtype dtype)))

(defun dtype-storage-only-p (dtype)
  (dtype-info-storage-only (find-dtype dtype)))

(defun dtype-zero (dtype)
  (dtype-info-zero (find-dtype dtype)))

(defun storage-element-type (storage)
  (etypecase storage
    (trivial-simd:vector-view
     (upgraded-array-element-type
      (dtype-element-type (trivial-simd:vector-view-type storage))))
    ((simple-array * (*)) (array-element-type storage))))

(defun storage-length (storage)
  (etypecase storage
    (trivial-simd:vector-view (trivial-simd:vector-view-length storage))
    ((simple-array * (*)) (length storage))))

(defun dtype-matches-storage-p (dtype storage)
  (equal (upgraded-array-element-type (dtype-element-type dtype))
         (storage-element-type storage)))

(defun storage-dtype (storage)
  "The computational dtype of STORAGE; 16-bit unsigned storage is :u16."
  (let ((match (find-if (lambda (info)
                          (and (not (dtype-info-storage-only info))
                               (dtype-matches-storage-p (dtype-info-name info) storage)))
                        *dtypes*)))
    (if match
        (dtype-info-name match)
        (error "No dtype for storage of element type ~S" (storage-element-type storage)))))
