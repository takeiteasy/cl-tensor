(in-package #:cl-tensor/tests)

(in-suite :cl-tensor)

(defvar *state*
  #+sbcl (sb-ext:seed-random-state 20261006)
  #-sbcl (make-random-state t))

(defun random-between (low high)
  (+ low (random (1+ (- high low)) *state*)))

(defun random-shape ()
  (loop repeat (random-between 0 4) collect (random-between 0 5)))

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

(defun reference-broadcast-shape (operands)
  (let* ((shapes (mapcar (lambda (x) (if (ct:tensorp x) (coerce (ct:tensor-shape x) 'list) nil)) operands))
         (rank (reduce #'max shapes :key #'length :initial-value 0)))
    (loop for axis from rank downto 1
          collect (let ((dimensions (remove-duplicates
                                     (remove 1 (mapcar (lambda (shape)
                                                         (if (< (length shape) axis) 1
                                                             (nth (- (length shape) axis) shape)))
                                                       shapes)))))
                    (when (> (length dimensions) 1) (error "Incompatible shapes"))
                    (or (first dimensions) 1)))))

(defun reference-broadcast-value (input indices)
  (if (ct:tensorp input)
      (let* ((shape (coerce (ct:tensor-shape input) 'list))
             (tail (nthcdr (- (length indices) (length shape)) indices))
             (local (mapcar (lambda (dimension index) (if (= dimension 1) 0 index)) shape tail)))
        (aref (ct:tensor-storage input)
              (reference-index (ct:tensor-offset input) (coerce (ct:tensor-strides input) 'list) local)))
      input))

(defun random-data-view (shape)
  (let* ((strides (loop repeat (length shape) collect (random-between -5 5)))
         (positions (mapcar (lambda (i) (reference-index 0 strides i)) (all-indices shape)))
         (low (reduce #'min positions :initial-value 0))
         (high (reduce #'max positions :initial-value 0))
         (storage (make-array (+ 3 (- high low)) :element-type 'single-float)))
    (dotimes (i (length storage)) (setf (aref storage i) (float (random-between 1 12) 1f0)))
    (ct:make-tensor-view storage shape :strides strides :offset (- 1 low))))

(defun random-output-view (shape)
  (let* ((strides (mapcar (lambda (stride) (* stride (if (zerop (random 2 *state*)) 1 -1)))
                          (reference-row-major-strides shape)))
         (positions (mapcar (lambda (i) (reference-index 0 strides i)) (all-indices shape)))
         (low (reduce #'min positions :initial-value 0))
         (high (reduce #'max positions :initial-value 0)))
    (ct:make-tensor-view (floats (+ 3 (- high low))) shape :strides strides :offset (- 1 low))))

(test random-broadcast-operations-match-reference
  (dotimes (trial 300)
    (let* ((target (random-shape))
           (a (random-data-view (loop for dimension in target
                                     collect (if (zerop (random 2 *state*)) 1 dimension))))
           (b (if (zerop (random 3 *state*)) 2f0
                  (random-data-view
                   (nthcdr (random-between 0 (length target))
                           (loop for dimension in target
                                 collect (if (zerop (random 2 *state*)) 1 dimension))))))
           (shape (reference-broadcast-shape (list a b)))
           (indices (all-indices shape)))
      (dolist (spec (list (list #'ct:add #'ct:add! #'+) (list #'ct:subtract #'ct:subtract! #'-)
                          (list #'ct:multiply #'ct:multiply! #'*) (list #'ct:divide #'ct:divide! #'/)
                          (list #'ct:min #'ct:min! #'min) (list #'ct:max #'ct:max! #'max)))
        (destructuring-bind (allocate write reference) spec
          (let* ((expected (mapcar (lambda (i) (funcall reference (reference-broadcast-value a i)
                                                       (reference-broadcast-value b i))) indices))
                 (result (funcall allocate a b))
                 (out (random-output-view shape)))
            (is (equalp (coerce shape 'vector) (ct:tensor-shape result)))
            (is (equalp expected (tensor-values result)))
            (is (eq out (funcall write out a b)))
            (is (equalp expected (tensor-values out)))
            (is (zerop (aref (ct:tensor-storage out) 0)))
            (is (zerop (aref (ct:tensor-storage out) (1- (length (ct:tensor-storage out)))))))))
      (let* ((expected (mapcar (lambda (i) (if (< (reference-broadcast-value a i)
                                                 (reference-broadcast-value b i)) 1 0)) indices))
             (mask (ct:compare :lt a b)))
        (is (equal expected (tensor-values mask)))
        (is (equalp (mapcar (lambda (i) (min (reference-broadcast-value a i)
                                           (reference-broadcast-value b i))) indices)
                    (tensor-values (ct:select mask a b)))))
      (is (equalp (mapcar (lambda (i) (reference-broadcast-value a i))
                         (all-indices (coerce (ct:tensor-shape a) 'list)))
                  (tensor-values (ct:copy-tensor a)))))))

(test random-output-injectivity-matches-reference
  (dotimes (trial 500)
    (let* ((out (random-data-view (random-shape)))
           (shape (coerce (ct:tensor-shape out) 'list))
           (indices (all-indices shape))
           (positions (mapcar (lambda (i)
                                (reference-index (ct:tensor-offset out)
                                                 (coerce (ct:tensor-strides out) 'list) i)) indices))
           (unique (= (length positions) (length (remove-duplicates positions))))
           (before (copy-seq (ct:tensor-storage out))))
      (if unique
          (progn (ct:add! out (ct:ones shape) 1)
                 (is (every (lambda (x) (= x 2)) (tensor-values out))))
          (progn (signals error (ct:add! out (ct:ones shape) 1))
                 (is (equalp before (ct:tensor-storage out))))))))
