# cl-tensor

NumPy-like N-dimensional tensors for Common Lisp, built on
[trivial-simd](https://git.sr.ht/~takeiteasy/trivial-simd) and `trivial-simd/blas`.

```lisp
(ql:quickload :cl-tensor)
(cl-tensor:make-tensor '(2 3))
;; => #<TENSOR :F32 (2 3)>
```

Construct, index and copy strided tensors; broadcast arithmetic, comparisons and selection; convert dtypes explicitly. See [limitations](docs/limitations.md) for planned operations.

## Docs

- [Design](docs/design.md)
- [Dtypes and strides](docs/dtypes-and-strides.md)
- [API](docs/api.md)
- [Constructors and access](docs/constructors.md)
- [Elementwise operations](docs/elementwise.md)
- [Conversion](docs/conversion.md)
- [Testing](docs/testing.md)
- [Limitations](docs/limitations.md)

Work is tracked at <https://todo.sr.ht/~takeiteasy/cl-tensor>.

## License

[MIT](LICENSE).
