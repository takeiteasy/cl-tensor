# Limitations

Constructors, access, copying, broadcasting operations, reductions, shape manipulation, real-float matrix multiplication and dtype conversion are implemented. The following areas remain open.

| Area | Ticket |
|---|---|
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
| Interleaved matmul destinations may use scratch or individual products | [#28](https://todo.sr.ht/~takeiteasy/cl-tensor/28) |
| Irregular reductions pack input layouts; reduction out-forms use output scratch | [#27](https://todo.sr.ht/~takeiteasy/cl-tensor/27) |
| Native integer/complex row reduction batching | [trivial-simd #148](https://todo.sr.ht/~takeiteasy/trivial-simd/148) |
| General output-layout validation uses O(element count) workspace when a sorted-stride uniqueness proof fails | [#25](https://todo.sr.ht/~takeiteasy/cl-tensor/25) |
| Mixed Lisp/foreign storage conservatively snapshots inputs, including disjoint memory | [#26](https://todo.sr.ht/~takeiteasy/cl-tensor/26) |
| Noncontiguous operations dispatch one bulk call per inner row; upstream stages strided rows | [#19](https://todo.sr.ht/~takeiteasy/cl-tensor/19) |

## trivial-simd gaps

| Missing upstream | cl-tensor ticket | trivial-simd ticket |
|---|---|---|
| Broader GEMM overlap and destination-layout proofs | [#28](https://todo.sr.ht/~takeiteasy/cl-tensor/28) | [#149](https://todo.sr.ht/~takeiteasy/trivial-simd/149) |
| N-d strided bulk operations | [#19](https://todo.sr.ht/~takeiteasy/cl-tensor/19) | [#146](https://todo.sr.ht/~takeiteasy/trivial-simd/146) |
| `log`, `tanh` and sigmoid kernel operators | [#20](https://todo.sr.ht/~takeiteasy/cl-tensor/20) | [#147](https://todo.sr.ht/~takeiteasy/trivial-simd/147) |

The umbrella ticket is [#9](https://todo.sr.ht/~takeiteasy/cl-tensor/9).
