(in-package #:cl-tensor/tests)

(in-suite :cl-tensor)

(defvar *state*
  #+sbcl (sb-ext:seed-random-state 20261006)
  #-sbcl (make-random-state t))

(defun random-between (low high)
  (+ low (random (1+ (- high low)) *state*)))

(defun random-shape ()
  (loop repeat (random-between 0 4) collect (random-between 0 5)))

(defun all-indices (shape)
  "Every index list of SHAPE in row-major order."
  (if (null shape)
      (list '())
      (loop for i below (first shape)
            nconc (mapcar (lambda (rest) (cons i rest)) (all-indices (rest shape))))))

(defun reference-row-major-strides (shape)
  (let ((stride 1))
    (reverse (loop for dimension in (reverse shape)
                   collect stride
                   do (setf stride (* stride dimension))))))

(defun reference-index (offset strides indices)
  (+ offset (reduce #'+ (mapcar #'* indices strides))))

(defun floats (n)
  (make-array n :element-type 'single-float :initial-element 0f0))

(defun view-or-nil (storage shape strides offset)
  (handler-case (ct:make-tensor-view storage shape :strides strides :offset offset)
    (error () nil)))

(test random-view-bounds-match-enumeration
  (dotimes (trial 2000)
    (let* ((shape (random-shape))
           (strides (loop repeat (length shape) collect (random-between -4 4)))
           (offset (random-between 0 10))
           (length (random-between 0 60))
           (accepted (view-or-nil (floats length) shape strides offset))
           (in-range (every (lambda (indices)
                              (< -1 (reference-index offset strides indices) length))
                            (all-indices shape))))
      (is (eq in-range (and accepted t))
          "shape ~S strides ~S offset ~D length ~D" shape strides offset length))))

(defun random-strides-for (shape)
  (if (zerop (random 2 *state*))
      (reference-row-major-strides shape)
      (loop repeat (length shape) collect (random-between -4 4))))

(test random-layout-predicates-match-enumeration
  (dotimes (trial 2000)
    (let* ((shape (random-shape))
           (strides (random-strides-for shape))
           (offset 100)
           (tensor (ct:make-tensor-view (floats 1000) shape :strides strides :offset offset))
           (indices (all-indices shape))
           (visited (mapcar (lambda (i) (ct:tensor-index tensor i)) indices)))
      (is (equal visited (mapcar (lambda (i) (reference-index offset strides i)) indices)))
      (is (eq (or (null visited)
                  (loop for index in visited for expected from offset always (= index expected)))
              (ct:contiguous-p tensor))
          "contiguous-p for shape ~S strides ~S" shape strides)
      (is (eq (or (null shape)
                  (<= (car (last shape)) 1)
                  (= 1 (car (last strides))))
              (ct:inner-contiguous-p tensor))
          "inner-contiguous-p for shape ~S strides ~S" shape strides))))

(test random-make-tensor-matches-reference
  (dotimes (trial 200)
    (let* ((shape (random-shape))
           (tensor (ct:make-tensor shape :dtype :f64)))
      (is (equalp (coerce (reference-row-major-strides shape) 'vector)
                  (ct:tensor-strides tensor)))
      (is (= (reduce #'* shape) (length (ct:tensor-storage tensor))))
      (is (equal (all-indices shape)
                 (loop for i from 0
                       for indices in (all-indices shape)
                       when (= i (ct:tensor-index tensor indices)) collect indices))))))
