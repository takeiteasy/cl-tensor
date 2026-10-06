(defpackage #:cl-tensor
  (:use #:cl)
  (:export #:*dtypes*
           #:dtype-element-type
           #:dtype-bytes
           #:dtype-storage-only-p
           #:storage-dtype
           #:tensor
           #:tensorp
           #:tensor-storage
           #:tensor-dtype
           #:tensor-shape
           #:tensor-strides
           #:tensor-offset
           #:tensor-rank
           #:tensor-size
           #:tensor-index
           #:row-major-strides
           #:make-tensor
           #:make-tensor-view
           #:contiguous-p
           #:inner-contiguous-p))
