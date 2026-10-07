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
| `tests/normalization.lisp` | Softmax, weighted RMSNorm, axis sets, widened sums, empty inputs, arbitrary layouts, aliases and seeded scalar references |
| `tests/matmul.lisp` | Vector/matrix and batch shapes, direct interleaved destinations, storage sharing and guards, batch dispatch counts, broadcast/reversed runs, empty contractions, aliases, foreign storage and seeded reference products |
| `tests/extensions.lisp` | External CLOS descriptors, operation tables, opaque block storage, mixed matmul, conversion, fallback, validation and aliases |
| `tests/random.lisp` | Seeded random shapes, strides, broadcasting and output uniqueness checked against independent Lisp calculations[^seed] |

## Backends and examples

The default run uses the selected trivial-simd backend. To check the portable Lisp backend:[^backend]

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:load-system :cl-tensor/tests)' \
  --eval '(let ((trivial-simd::*backend* :lisp)) (asdf:test-system :cl-tensor))'
```

Run the constructor, broadcasting, activation, normalization, shape, reduction and matmul examples:

```sh
sbcl --non-interactive --load examples/tensors.lisp
```

Run the external dtype/storage example:

```lisp
(load "examples/extensions.lisp")
(cl-tensor/extension-example:run-example)
```

See [extensions](extensions.md) for its public protocol and [dispatch measurements](extensions-performance.md) for the regression comparison.

Measure elementwise time and, on SBCL, allocation:

```sh
sbcl --non-interactive --load tests/elementwise-bench.lisp
```

See [elementwise measurements](elementwise-performance.md) for layouts, recorded results and baseline comparison.

Compare complete normalization calls using typed Lisp and existing kernels:

```sh
sbcl --non-interactive --load tests/normalization-bench.lisp
```

See [normalization measurements](normalization-performance.md) for validation, path selection and retained results.

Measure interleaved matrix and batch destinations, complete-call time and Lisp allocation:

```sh
sbcl --dynamic-space-size 4096 --non-interactive --load tests/matmul-bench.lisp
```

See [matmul measurements](matmul-performance.md) for layouts, baseline comparison and retained results.

## Extension protocol validation

SBCL 2.6.8 and ECL execute the full suite and the constructor and extension examples. Recorded run, 2026-10-07:

| Implementation/backend | Passing checks |
|---|---:|
| SBCL default/native | 46,909 |
| SBCL Lisp | 46,879 |
| ECL | 45,988 |
| Experimental ARM64 CCL | 31,358; subject to the [CCL limitation](#limitations) |

The extension checks cover public-interface-only external packages, opaque copying and writes, readable-storage fallback, conversion destination selection, malformed selectors, and validation before execution. See [dispatch measurements](extensions-performance.md) for timing and allocation.

## Limitations

- The experimental ARM64 CCL `1.13 (v1.13-459-g690ff7ea)` build can crash or end compiled normalization reference checks prematurely while reporting success. Its green summary does not establish complete normalization coverage: [#31](https://todo.sr.ht/~takeiteasy/cl-tensor/31). SBCL native/Lisp and ECL execute the full checks.

[^seed]: SBCL runs use a fixed seed, so failures reproduce. Other Lisps seed from the clock.
[^backend]: The backend binding is an upstream test hook, not a cl-tensor public API. Use `:native` only when the native library is available; ARM64 does not provide the `:sbcl` SIMD backend.
