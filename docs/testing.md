# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-tensor)'
```

`asdf:test-system` loads `cl-tensor/tests` and signals an error if any check fails. From a REPL, `(cl-tensor/tests:run-tests)` returns true when every check passes.

| File | Covers |
|---|---|
| `tests/tensor.lisp` | Dtype inference, construction, view validation, indexing |
| `tests/constructors.lisp` | Filled tensors, nested data, sequences, identity matrices, access, copying and foreign storage |
| `tests/elementwise.lisp` | Arithmetic, real-float activations, masks, conversion, dtype errors, strided output, aliases across axes, and one N-D dispatch per operation |
| `tests/shape.lisp` | View sharing, reshape copying, slice bounds, joins/splits and seeded arbitrary layouts |
| `tests/reductions.lisp` | All reducers, dtypes, axes, widening, empty groups, aliases, foreign storage and seeded enumeration |
| `tests/matmul.lisp` | Vector/matrix and batch shapes, stride-aware storage sharing, batch dispatch counts, broadcast/reversed runs, scratch fallbacks, empty contractions, aliases, foreign storage and seeded reference products |
| `tests/random.lisp` | Seeded random shapes, strides, broadcasting and output uniqueness checked against independent Lisp calculations[^seed] |

## Backends and examples

The default run uses the selected trivial-simd backend. To check the portable Lisp backend:[^backend]

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:load-system :cl-tensor/tests)' \
  --eval '(let ((trivial-simd::*backend* :lisp)) (asdf:test-system :cl-tensor))'
```

Run the constructor, broadcasting, activation, shape, reduction and matmul examples:

```sh
sbcl --non-interactive --load examples/tensors.lisp
```

Measure elementwise time and, on SBCL, allocation:

```sh
sbcl --non-interactive --load tests/elementwise-bench.lisp
```

See [elementwise measurements](elementwise-performance.md) for layouts, recorded results and baseline comparison.

[^seed]: SBCL runs use a fixed seed, so failures reproduce. Other Lisps seed from the clock.
[^backend]: The backend binding is an upstream test hook, not a cl-tensor public API. Use `:native` only when the native library is available; ARM64 does not provide the `:sbcl` SIMD backend.
