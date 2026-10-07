# Elementwise operations

Operations broadcast trailing axes and return independent contiguous tensors. A matching `!` form writes into a supplied output and returns it, for example `(add a b)` and `(add! out a b)`.


The dtype rules below describe built-in implementations. [Extension methods](extensions.md) select accepted dtype combinations and result dtypes before built-in execution.

## Operations

| Allocating call | Result | Dtypes |
|---|---|---|
| `(add a b)`, `(subtract a b)`, `(multiply a b)`, `(divide a b)` | Arithmetic | Real, complex, integer |
| `(negate input)`, `(abs input)` | Negation, absolute value | Real, complex, integer |
| `(sqrt input)`, `(reciprocal input)` | Square root, reciprocal | Real, complex floating-point |
| `(log input)`, `(tanh input)`, `(sigmoid input)` | Natural logarithm, hyperbolic tangent, logistic sigmoid | `:f32`, `:f64` |
| `(min a b)`, `(max a b)`, `(clamp input lower upper)` | Bounds | Real, integer |
| `(compare operator a b)` | `:u8` mask containing 0 or 1 | Equality for all computational dtypes; ordering for real/integer |
| `(select mask on-true on-false &key dtype)` | Element selected by mask | Real, complex, integer |

Out-forms prepend `out` to these arguments; `select!` has no `:dtype` keyword. Comparison operators are `:eq`, `:ne`, `:lt`, `:le`, `:gt`, `:ge`. Selection requires a `:u8` tensor mask: zero is false and any nonzero byte is true.

Complex `abs` returns `:f32` for `:c32` input or `:f64` for `:c64` input. Other numeric results retain the input dtype. Use package-qualified names for `abs`, `sqrt`, `min`, `max`, `log` and `tanh`, which shadow Common Lisp symbols.

## Real-float activations

`log` takes one input and computes the natural logarithm. `tanh` and `sigmoid`
retain the input dtype and shape. Their `!` forms follow the same output and
alias rules as other elementwise operations.

```lisp
(ct:sigmoid (ct:from-data '(-1 0 1)))
;; approximately (0.26894143 0.5 0.7310586)
(ct:multiply x (ct:sigmoid x)) ; SiLU
```

These operations use scalar system math, without a fixed ULP bound or
cross-backend bit identity. Positive finite inputs have real logarithms.
Nonpositive logarithm inputs, exceptional values, underflow and floating-point
traps follow the active backend and Lisp environment. Sigmoid uses a
sign-dependent formula that avoids exponential overflow.[^activation]

## Broadcasting and scalars

Align shapes from the right. Each pair of dimensions must match or one must be 1; a missing axis behaves as 1. Broadcasting 0 with 1 produces 0.

```lisp
(let ((a (ct:from-data '((1) (2))))
      (b (ct:from-data '(10 20 30))))
  (ct:add a b))
;; => #<TENSOR :F32 (2 3)> with rows (11 21 31), (12 22 32)
```

Lisp scalars are accepted on either side and coerced to the numeric tensor dtype. Numeric tensors must have identical dtypes; convert explicitly with [astype](conversion.md). Allocating arithmetic needs a numeric tensor to determine dtype. For selection with two scalar choices, supply `:dtype`:

```lisp
(ct:select (ct:compare :gt (ct:arange 4) 1) 10 0 :dtype :s16)
;; values: 0, 0, 10, 10
```

Out-forms use the destination dtype when all numeric operands are scalars.[^scalar-out] Half-float storage must be converted before arithmetic. Integer arithmetic wraps at the dtype width, and division truncates toward zero; zero integer divisors signal `division-by-zero`.

## Output and aliases

Output shape and dtype must exactly match the result. Reversed, transposed and padded outputs work; outputs that map several logical elements to one storage location signal an error before writes.

Inputs are read as if copied before overlapping output writes, including aliases across rows and distinct foreign views of shared memory. Exact elementwise in-place mappings need no snapshot. Ordinary validation happens before writes, even for empty outputs; arithmetic errors may leave part of the output updated.[^execution]

[^scalar-out]: Scalars have shape `nil`, so scalar-only arithmetic requires a rank-zero output. `select!` also broadcasts its mask. Scalar-only `compare!` uses its `:u8` output as numeric context, so operands must be in-range integers.
[^execution]: Each operation uses one upstream N-D call. Native-supported operations traverse signed and broadcast strides in C, using bulk kernels where available and scalar system math for activations. Complex operations, remaining conversion pairs and fallback configurations traverse storage directly in Lisp. See [performance limitations](limitations.md#performance) for snapshot and layout-validation workspace costs. Exceptional floating-point behavior follows trivial-simd and the active Lisp floating-point environment.

[^activation]: Sigmoid computes `z = exp(-abs(x))`, then `1/(1+z)` for nonnegative inputs or `z/(1+z)` otherwise. Underflow remains backend-dependent. See [examples](../examples/tensors.lisp) for composed SiLU and tanh-approximate GELU, and [normalization](normalization.md) for softmax and RMSNorm.
