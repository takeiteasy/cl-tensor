# cl-tensor

NumPy-like N-dimensional tensors for Common Lisp, built on
[trivial-simd](https://git.sr.ht/~takeiteasy/trivial-simd) and `trivial-simd/blas`.

```lisp
(ql:quickload :cl-tensor)
(cl-tensor:make-tensor '(2 3))
;; => #<TENSOR :F32 (2 3)>
```

Tensors with shape, strides and views exist today. Elementwise operations, reductions and matmul are planned.

## Docs

- [Design](docs/design.md)
- [Dtypes and strides](docs/dtypes-and-strides.md)
- [API](docs/api.md)
- [Testing](docs/testing.md)
- [Limitations](docs/limitations.md)

Work is tracked at <https://todo.sr.ht/~takeiteasy/cl-tensor>.

## License

[MIT](LICENSE).
