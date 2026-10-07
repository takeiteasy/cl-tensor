# Limitations

Constructors, access, copying, broadcasting operations, reductions, softmax, RMSNorm, shape manipulation, real-float matrix multiplication and dtype conversion are implemented. The following areas remain open.

| Area | Ticket |
|---|---|
| Dtype and storage extension protocol | [#23](https://todo.sr.ht/~takeiteasy/cl-tensor/23) |
| Quantized dtypes, GGUF reader and model executor (planned `cl-inference` repo) | [#24](https://todo.sr.ht/~takeiteasy/cl-tensor/24) |
| Kernel fusion | [#12](https://todo.sr.ht/~takeiteasy/cl-tensor/12) |
| Parallel operations | [#13](https://todo.sr.ht/~takeiteasy/cl-tensor/13) |
| Autograd | [#14](https://todo.sr.ht/~takeiteasy/cl-tensor/14) |
| Continuous integration | [#22](https://todo.sr.ht/~takeiteasy/cl-tensor/22) |
| Complete normalization validation on experimental ARM64 CCL | [#31](https://todo.sr.ht/~takeiteasy/cl-tensor/31) |

## Performance

| Limitation | Ticket |
|---|---|
| Short elementwise calls pay fixed N-D layout setup overhead | [trivial-simd #152](https://todo.sr.ht/~takeiteasy/trivial-simd/152) |
| Exact GEMM layout proofs can require substantial search time for difficult layouts | [trivial-simd #154](https://todo.sr.ht/~takeiteasy/trivial-simd/154) |
| Irregular reductions pack input layouts; reduction out-forms use output scratch | [#27](https://todo.sr.ht/~takeiteasy/cl-tensor/27) |
| Normalization uses full-result scratch; portable f64 weight/output stages allocate per element, and restoring axis order can require another tensor | [#30](https://todo.sr.ht/~takeiteasy/cl-tensor/30) |
| Native integer/complex row reduction batching | [trivial-simd #148](https://todo.sr.ht/~takeiteasy/trivial-simd/148) |
| General output-layout validation uses O(element count) workspace when a sorted-stride uniqueness proof fails | [#25](https://todo.sr.ht/~takeiteasy/cl-tensor/25), [trivial-simd #150](https://todo.sr.ht/~takeiteasy/trivial-simd/150) |
| Mixed Lisp/foreign storage conservatively snapshots inputs, including disjoint memory | [#26](https://todo.sr.ht/~takeiteasy/cl-tensor/26), [trivial-simd #151](https://todo.sr.ht/~takeiteasy/trivial-simd/151) |

The umbrella ticket is [#9](https://todo.sr.ht/~takeiteasy/cl-tensor/9).
