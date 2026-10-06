# Limitations

Constructors, element access, copying, broadcasting operations and dtype conversion are implemented. The following areas remain open.

| Area | Ticket |
|---|---|
| Axis reductions | [#6](https://todo.sr.ht/~takeiteasy/cl-tensor/6) |
| Reshape, transpose, slicing, concatenation | [#7](https://todo.sr.ht/~takeiteasy/cl-tensor/7) |
| `matmul` and batched `matmul` | [#8](https://todo.sr.ht/~takeiteasy/cl-tensor/8) |
| Dtype and storage extension protocol | [#23](https://todo.sr.ht/~takeiteasy/cl-tensor/23) |
| Quantized dtypes, GGUF reader and model executor (planned `cl-inference` repo) | [#24](https://todo.sr.ht/~takeiteasy/cl-tensor/24) |
| Softmax and RMSNorm | [#21](https://todo.sr.ht/~takeiteasy/cl-tensor/21) |
| Kernel fusion | [#12](https://todo.sr.ht/~takeiteasy/cl-tensor/12) |
| Parallel operations | [#13](https://todo.sr.ht/~takeiteasy/cl-tensor/13) |
| Autograd | [#14](https://todo.sr.ht/~takeiteasy/cl-tensor/14) |
| Continuous integration | [#22](https://todo.sr.ht/~takeiteasy/cl-tensor/22) |

## Performance

| Limitation | Ticket |
|---|---|
| General output-layout validation uses O(element count) workspace when a sorted-stride uniqueness proof fails | [#25](https://todo.sr.ht/~takeiteasy/cl-tensor/25) |
| Mixed Lisp/foreign storage conservatively snapshots inputs, including disjoint memory | [#26](https://todo.sr.ht/~takeiteasy/cl-tensor/26) |
| Noncontiguous operations dispatch one bulk call per inner row; upstream stages strided rows | [#19](https://todo.sr.ht/~takeiteasy/cl-tensor/19) |

## trivial-simd gaps

| Missing upstream | cl-tensor ticket | trivial-simd ticket |
|---|---|---|
| Strided-batched GEMM and arbitrary-stride matrix views | [#17](https://todo.sr.ht/~takeiteasy/cl-tensor/17) | [#144](https://todo.sr.ht/~takeiteasy/trivial-simd/144) |
| Row/axis reductions and `prod` | [#18](https://todo.sr.ht/~takeiteasy/cl-tensor/18) | [#145](https://todo.sr.ht/~takeiteasy/trivial-simd/145) |
| N-d strided bulk operations | [#19](https://todo.sr.ht/~takeiteasy/cl-tensor/19) | [#146](https://todo.sr.ht/~takeiteasy/trivial-simd/146) |
| `log`, `tanh` and sigmoid kernel operators | [#20](https://todo.sr.ht/~takeiteasy/cl-tensor/20) | [#147](https://todo.sr.ht/~takeiteasy/trivial-simd/147) |

The umbrella ticket is [#9](https://todo.sr.ht/~takeiteasy/cl-tensor/9).
