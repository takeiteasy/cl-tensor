# Limitations

The tensor type exists. Operations on it do not yet.

| Area | Ticket |
|---|---|
| Constructors (`zeros`, `arange`, …), element access, printing, `copy-tensor` | [#4](https://todo.sr.ht/~takeiteasy/cl-tensor/4) |
| Broadcasting elementwise operations, `astype`, scalar coercion | [#5](https://todo.sr.ht/~takeiteasy/cl-tensor/5) |
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

## trivial-simd gaps

| Missing upstream | cl-tensor ticket | trivial-simd ticket |
|---|---|---|
| Strided-batched GEMM and arbitrary-stride matrix views | [#17](https://todo.sr.ht/~takeiteasy/cl-tensor/17) | [#144](https://todo.sr.ht/~takeiteasy/trivial-simd/144) |
| Row/axis reductions and `prod` | [#18](https://todo.sr.ht/~takeiteasy/cl-tensor/18) | [#145](https://todo.sr.ht/~takeiteasy/trivial-simd/145) |
| N-d strided bulk operations | [#19](https://todo.sr.ht/~takeiteasy/cl-tensor/19) | [#146](https://todo.sr.ht/~takeiteasy/trivial-simd/146) |
| `log`, `tanh` and sigmoid kernel operators | [#20](https://todo.sr.ht/~takeiteasy/cl-tensor/20) | [#147](https://todo.sr.ht/~takeiteasy/trivial-simd/147) |

The umbrella ticket is [#9](https://todo.sr.ht/~takeiteasy/cl-tensor/9).
