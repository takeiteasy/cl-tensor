# cl-tensor

NumPy-like N-dimensional tensors for Common Lisp, built on
[trivial-simd](https://git.sr.ht/~takeiteasy/trivial-simd) and `trivial-simd/blas`.

Early bootstrap: the system and package exist, with no tensor API yet.

```lisp
(ql:quickload :cl-tensor)
```

Planned scope is tensors with shape and strides, broadcasting, axis reductions
and matmul. Work is tracked at <https://todo.sr.ht/~takeiteasy/cl-tensor>.

## License

[MIT](LICENSE).
