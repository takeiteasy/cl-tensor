(require :asdf)

(handler-bind
    ((warning (lambda (condition)
                (when (search "Deprecated recursive use" (princ-to-string condition)
                              :test #'char-equal)
                  (error "Recursive ASDF loading: ~A" condition)))))
  (asdf:test-system :cl-tensor)
  (asdf:test-system :cl-tensor :force t))
