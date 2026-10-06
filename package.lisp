(defpackage #:cl-tensor
  (:use #:cl)
  (:shadow #:abs #:sqrt #:min #:max)
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
           #:inner-contiguous-p
           #:zeros #:ones #:full #:from-data #:arange #:linspace #:eye
           #:tref #:copy-tensor #:astype
           #:add #:add! #:subtract #:subtract! #:multiply #:multiply! #:divide #:divide!
           #:negate #:negate! #:abs #:abs! #:sqrt #:sqrt! #:reciprocal #:reciprocal!
           #:min #:min! #:max #:max! #:clamp #:clamp!
           #:compare #:compare! #:select #:select!))
