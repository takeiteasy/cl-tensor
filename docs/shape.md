# Shape operations

Shape operations create views sharing storage. `concatenate` and `stack` allocate independent contiguous results; `reshape` copies only when its copy policy requires it.

| Call | Behavior |
|---|---|
| `(reshape input shape &key (copy :auto))` | Preserve logical row-major element order |
| `(transpose input &key axes)` | Permute every axis; default reverses axis order |
| `(slice input &key selectors)` | Basic indexing and ranges as a view |
| `(squeeze input &key axis)` | Remove singleton axes; default removes all |
| `(expand-dims input &key axis)` | Insert one singleton axis; axis is required |
| `(concatenate tensors &key (axis 0))` | Join along an existing axis |
| `(stack tensors &key (axis 0))` | Join along a new axis |
| `(split input sections-or-cuts &key (axis 0))` | Return a list of views along an existing axis |

Negative axes count from the end. For `stack` and `expand-dims`, axes refer to the resulting rank. All operations preserve dtype, including raw `:f16` and `:bf16` storage.

[Storage extensions](extensions.md) validate each resulting view and may reject unsupported layouts. Copying paths use the dtype copy method, allowing opaque storage when its layout supports the operation.

## Reshape and transpose

A reshape shape contains non-negative dimensions and at most one inferred `-1`. Its element count must match the input. Inference with a zero known product is ambiguous and signals an error.

| Copy policy | Result |
|---|---|
| `:auto` | Share storage when strides allow; otherwise copy |
| `:never` | Share storage or signal an error |
| `:always` | Allocate independent storage |

Compatible padded, stepped, reversed and zero-stride layouts can reshape without a copy.[^strides] Transpose requires a permutation of all axes.

```lisp
(let* ((a (ct:reshape (ct:arange 6) '(2 3)))
       (b (ct:transpose a)))
  (ct:reshape b '(6)))
;; values: 0, 3, 1, 4, 2, 5; independent storage
```

## Slicing

`:selectors` is a list or vector with one selector per supplied axis. Omitted trailing axes are retained.

| Selector | Meaning |
|---|---|
| `:all` | Retain the complete axis |
| Integer | Select one element and drop the axis; negative indices count from the end |
| `(start stop)` | Range with exclusive stop and step 1 |
| `(start stop step)` | Range with a nonzero signed step |

Bounds are clipped to the axis. Negative bounds count from the end. `nil` endpoints select the beginning/end appropriate to the step direction. For reverse ranges, an omitted stop differs from an explicit `-1`.

```lisp
(let ((a (ct:reshape (ct:arange 12) '(3 4))))
  (ct:slice a :selectors '(1 (nil nil -1))))
;; shape: (4); values: 7, 6, 5, 4; shared storage
```

Integer indices outside the axis and excess selectors signal errors. Selecting every axis by integer produces a rank-zero tensor; use `tref` to read its scalar.

## Singleton axes and joining

`squeeze :axis` accepts an integer or sequence; every selected dimension must be one. An empty vector removes none. `expand-dims` accepts one axis.

Concatenate requires matching dtypes and all dimensions outside the join axis. Stack requires identical shapes and dtypes. Both accept a nonempty sequence of tensors, including strided views.

Split accepts a positive section count dividing the axis evenly, or ordered cut points between zero and the axis size. Repeated cuts create empty views. Uneven splits use explicit cuts.

```lisp
(mapcar #'ct:tensor-shape (ct:split (ct:arange 7) '(2 5)))
;; => (#(2) #(3) #(2))
```

## Limitations

- Slicing supports basic integer and range selectors. Boolean masks, advanced array indexing, ellipsis and insertion selectors are outside this API.

[^strides]: Reshape groups compatible stride chunks, ignoring singleton-axis strides. A new dimension cannot cross a discontinuity between chunks. Views retain their storage owner; callers keep foreign memory valid for the view's lifetime.
