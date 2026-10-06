# Design

A tensor is typed storage plus a strided window onto it. Views share storage; constructors, `copy-tensor`, `astype` and allocating operations return independent storage.

| Slot | Meaning |
|---|---|
| `storage` | Simple vector, or a trivial-simd `vector-view` |
| `dtype` | Element type keyword, see [dtypes and strides](dtypes-and-strides.md) |
| `shape` | Size of each axis |
| `strides` | Storage step per axis, in elements |
| `offset` | Storage index of the first element |

The element at indices `i0 … in` lives at `offset + Σ ik · stridek`.

```lisp
(let* ((storage (make-array 6 :element-type 'single-float))
       (a (ct:make-tensor-view storage '(2 3)))
       (b (ct:make-tensor-view storage '(3 2) :strides '(1 3))))
  (eq (ct:tensor-storage a) (ct:tensor-storage b)))
;; => T
```

## Tensor type

`tensor` is a structure with read-only slots.[^struct] Shape and strides are `(simple-array fixnum (*))`. `make-tensor` allocates zeroed contiguous storage; `make-tensor-view` wraps existing storage and validates it.

A view is valid when:

- the strides match the rank, the offset is non-negative and every size is non-negative;
- the dtype matches the storage element type;
- every reachable storage index lies inside the storage. An empty view is always valid.

## Extension points

Autograd attaches through a wrapper around a tensor, or through a subtype defined with `(:include tensor)`. Code that accepts a `tensor` accepts both.

## Shape and reduction execution

Shape operations build validated views; reshape copies when its strides cannot represent the requested logical order. Concatenate and stack allocate independent storage.

Reductions group retained and reduced axes into rows, packing when inner rows are not contiguous. Compatible real float batches execute natively; integer and complex batches use upstream typed reducers. Destination-writing reductions compute before scattering, so overlapping views read their original values.

See [reductions](reductions.md) and [shape operations](shape.md) for their interfaces.

## Operation execution

Elementwise operations compute the broadcast shape, validate dtypes and output layout, snapshot aliased inputs, then dispatch contiguous tensors or inner rows to trivial-simd. Inputs support arbitrary strides. Out-forms reject layouts in which two logical elements share a storage location.

Lisp storage uses array access. Foreign storage uses typed CFFI access and remains owned by the caller; its memory must stay valid for the entire operation.

See [limitations](limitations.md) for what is not built yet.

[^struct]: Accessors are `tensor-storage`, `tensor-dtype`, `tensor-shape`, `tensor-strides` and `tensor-offset`. The shape and stride vectors are not copied on access; do not modify them.
