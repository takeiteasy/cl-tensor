# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(asdf:test-system :cl-tensor)'
```

`asdf:test-system` loads `cl-tensor/tests` and signals an error if any check fails. From a REPL, `(cl-tensor/tests:run-tests)` returns true when every check passes.

| File | Covers |
|---|---|
| `tests/tensor.lisp` | Dtype inference, construction, view validation, indexing |
| `tests/random.lisp` | Seeded random shapes, strides and offsets checked against plain Lisp loops[^seed] |

[^seed]: SBCL runs use a fixed seed, so failures reproduce. Other Lisps seed from the clock.
