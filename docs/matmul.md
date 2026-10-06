# Matrix multiplication

`matmul` multiplies vectors, matrices and batches into an independent contiguous tensor. `(matmul! out a b)` writes the result into `out` and returns it. Both operands must be tensors with matching `:f32` or `:f64` dtypes; use [astype](conversion.md) for explicit conversion.

## Shapes

The final two axes are matrix axes; leading axes broadcast using the [elementwise broadcasting rules](elementwise.md#broadcasting-and-scalars). Contraction dimensions must match. Scalars and rank-zero tensors signal errors.

| Left shape | Right shape | Result shape |
|---|---|---|
| `(K)` | `(K)` | `()` — rank-zero tensor |
| `(M K)` | `(K)` | `(M)` |
| `(K)` | `(K N)` | `(N)` |
| `(M K)` | `(K N)` | `(M N)` |
| `(2 1 M K)` | `(1 3 K N)` | `(2 3 M N)` |
| `(K)` | `(2 K N)` | `(2 N)` |

A left vector behaves as a `(1 K)` matrix and a right vector as a `(K 1)` matrix; the inserted axes disappear from the result.

```lisp
(let ((a (ct:from-data '((1 2 3) (4 5 6))))
      (b (ct:from-data '(2 3 4))))
  (ct:matmul a b))
;; => #<TENSOR :F32 (2)> with values (20 47)

(ct:tref (ct:matmul (ct:from-data '(1 2 3)) (ct:from-data '(4 5 6))))
;; => 32f0
```

## Destinations and empty dimensions

The destination must match the result shape and dtype exactly. Padded, transposed and reversed destinations work; layouts mapping multiple logical elements to one location signal errors before writes. Overlapping inputs are read as if copied before any destination write, including aliases across batches and foreign views.[^output]

```lisp
(let ((out (ct:transpose (ct:zeros '(3 2)))))
  (ct:matmul! out (ct:ones '(2 4)) (ct:ones '(4 3))))
;; => OUT, shape (2 3), every value is 4f0
```

An empty output performs no writes but still validates operands, shapes and dtype. A zero contraction dimension produces typed zeros, including a zero for an empty vector dot product.

## Execution

Vectors use BLAS dot and matrix-vector routines; matrix products use stride-aware GEMM views over their backing storage. Transposed, reversed, padded and zero-stride input matrices work directly. Compatible batch axes combine into constant-stride runs, with one batched GEMM call per run; broadcast operands use a zero batch stride. A single product uses ordinary GEMM.[^backend]

Matrix-vector products retain transpose flags and pack irregular matrices into reusable buffers. Zero-stride vectors materialize. Valid irregular destinations that fail upstream's uniqueness proof use reusable output scratch; batches that fail the combined proof dispatch individual products. See [limitations](#limitations).

See [performance limitations](limitations.md#performance) for workspace costs shared with other operations.

## Limitations

- Interleaved destination layouts may require output scratch or individual products because upstream uses a conservative uniqueness proof. Stronger validation is tracked in [#28](https://todo.sr.ht/~takeiteasy/cl-tensor/28).

[^output]: Every overlapping input is snapshotted before any batch writes. Destinations proven unique by upstream receive direct writes; other valid matrix layouts use one reusable matrix buffer before scattering. Ordinary validation precedes writes; computation errors may leave part of the destination updated.
[^backend]: Floating-point order and exceptional values follow trivial-simd and the active Lisp floating-point environment; final bits may differ between backends. BLAS kernel dimensions, leading dimensions and vector increments are bounded by 1,073,741,823, and accessed storage offsets by 2⁶⁰−1. Irrelevant singleton-axis strides are ignored. GEMM uses signed row/column strides; input strides outside kernel limits pack into reusable scratch. Batch runs split at the upstream count limit. Native batching requires the upstream batch symbols; older native libraries fall back to per-product dispatch.
