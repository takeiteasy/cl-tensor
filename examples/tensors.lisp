(require :asdf)
(asdf:load-system :cl-tensor)

(defpackage #:cl-tensor/examples
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor)))

(in-package #:cl-tensor/examples)

(let ((sequence (ct:arange 5 0 -2 :dtype :s8))
      (samples (ct:linspace 0 1 :num 3))
      (identity (ct:eye 2 :columns 3 :k 1)))
  (assert (= 3 (ct:tref sequence 1)))
  (assert (= 0.5f0 (ct:tref samples 1)))
  (assert (= 1f0 (ct:tref identity 1 2))))

(let* ((rows (ct:from-data '((1) (2))))
       (columns (ct:from-data '(10 20 30)))
       (sum (ct:add rows columns))
       (mask (ct:compare :gt sum 20))
       (selected (ct:select mask sum 0))
       (out (ct:zeros '(2 3))))
  (assert (= 32f0 (ct:tref sum 1 2)))
  (assert (= 0f0 (ct:tref selected 0 0)))
  (ct:multiply! out selected 2)
  (assert (= 64f0 (ct:tref out 1 2)))
  (let ((copy (ct:copy-tensor out)))
    (setf (ct:tref copy 0 0) 7)
    (assert (= 0f0 (ct:tref out 0 0))))
  (assert (= 64 (ct:tref (ct:astype out :s16) 1 2))))

(let* ((values (ct:from-data '(1 2 3)))
       (bits (ct:astype values :bf16)))
  (assert (= 2f0 (ct:tref (ct:astype bits :f32) 1))))

(format t "Constructor, broadcasting and conversion examples passed.~%")

(let* ((matrix (ct:reshape (ct:arange 12) '(3 4)))
       (reversed (ct:slice matrix :selectors '(:all (nil nil -1))))
       (means (ct:mean reversed :axis 1 :keepdims t))
       (centered (ct:subtract reversed means)))
  (format t "~&Reversed rows: ~S~%Row means: ~S~%Centered row sums: ~S~%"
          (ct:tensor-shape reversed) (ct:tensor-storage means)
          (ct:tensor-storage (ct:sum centered :axis 1))))
