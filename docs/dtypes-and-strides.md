# Dtypes and strides

Dtypes are strict and strides are arbitrary.

## Dtypes

| Dtype | Element type | Bytes | Computable |
|---|---|---|---|
| `:f32` `:f64` | `single-float` `double-float` | 4, 8 | yes |
| `:c32` `:c64` | complex single/double | 8, 16 | yes |
| `:s8` `:s16` `:s32` `:s64` | `(signed-byte n)` | n/8 | yes |
| `:u8` `:u16` `:u32` `:u64` | `(unsigned-byte n)` | n/8 | yes |
| `:f16` `:bf16` | `(unsigned-byte 16)` | 2 | storage only |

- Mixed dtypes signal an error. Conversion is explicit, with `astype`.
- Scalars are coerced to the tensor dtype. A float scalar with an integer tensor signals an error.
- Storage of `(unsigned-byte 16)` infers as `:u16`. Pass `:f16` or `:bf16` explicitly:

```lisp
(let ((bits (make-array 4 :element-type '(unsigned-byte 16))))
  (list (ct:tensor-dtype (ct:make-tensor-view bits '(4)))
        (ct:tensor-dtype (ct:make-tensor-view bits '(4) :dtype :bf16))))
;; => (:U16 :BF16)
```

Dtype conversion and scalar coercion are listed in [limitations](limitations.md).

## Strides

Strides count elements, not bytes, and may be negative or zero.

| Stride | Gives |
|---|---|
| positive | forward walk |
| negative | reversed axis (give a matching `:offset`) |
| zero | broadcast axis, every index reads the same element |

```lisp
(let ((v (ct:make-tensor-view (make-array 6 :element-type 'single-float) '(2 3)
                              :strides '(-3 1) :offset 3)))
  (list (ct:tensor-index v '(0 0)) (ct:tensor-index v '(1 0))))
;; => (3 0)
```

## Layout predicates

| Function | True when |
|---|---|
| `contiguous-p` | elements occupy consecutive storage in row-major order |
| `inner-contiguous-p` | the innermost axis has unit stride |

Axes of size 1 ignore their stride.[^fast]

[^fast]: Operations take the kernel fast path when `inner-contiguous-p` holds and loop over the outer axes. Other operands are copied to a contiguous temporary first.
