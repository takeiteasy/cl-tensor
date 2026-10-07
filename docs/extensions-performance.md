# Extension dispatch measurements

Dtype selection adds about 0.19 µs to a short native addition and about 0.15–0.20 µs to tiny native matmul calls on this Apple M1. The larger cases in the table remain within 2.3% of their baseline times.[^measurement]

## Complete-call comparison

SBCL 2.6.8, Apple M1, 2026-10-07. Baseline is commit `c34b0ba`; protocol results include registry lookup, CLOS selection and unchanged built-in backend execution. Inputs and destinations are ordinary preallocated arrays, so these calls do not exercise custom-storage packing or extension output staging.

| Native operation | Baseline µs/call | Protocol µs/call | Baseline bytes/call | Protocol bytes/call |
|---|---:|---:|---:|---:|
| Add, 1,024 contiguous f32 elements | 1.816 | 2.010 | 1,677 | 1,887 |
| Add, 65,536 contiguous f32 elements | 10.032 | 10.259 | 1,654 | 1,867 |
| Add, 65,536 transposed f32 elements | 87.123 | 87.494 | 1,720 | 1,933 |
| Add, 65,536 broadcast f32 elements | 16.052 | 16.326 | 1,720 | 1,933 |
| Matmul, f32 matrix 3×2 | 3.615 | 3.810 | 2,620 | 3,276 |
| Matmul, f64 matrix 3×2 | 3.694 | 3.848 | 2,620 | 3,276 |
| Matmul, 256 interleaved f32 matrices | 129.641 | 130.951 | 141,315 | 141,453 |
| Matmul, 256 interleaved f64 batches | 98.506 | 98.898 | 106,448 | 106,630 |
| Matmul, f64 contiguous batch control | 23.090 | 23.076 | 2,620 | 3,276 |

Values are medians of three serial fresh-process trials. Retained results: [baseline 1](benchmark-runs/2026-10-07-extensions-before-1.txt), [2](benchmark-runs/2026-10-07-extensions-before-2.txt), [3](benchmark-runs/2026-10-07-extensions-before-3.txt); [protocol 1](benchmark-runs/2026-10-07-extensions-after-1.txt), [2](benchmark-runs/2026-10-07-extensions-after-2.txt), [3](benchmark-runs/2026-10-07-extensions-after-3.txt). Each includes portable Lisp matmul results and the source directory actually loaded.

## Reproduce

Run the existing benchmark scripts in separate fresh processes for each trial:

```sh
sbcl --non-interactive --load tests/elementwise-bench.lisp \
  --load tests/matmul-bench.lisp
```

For a baseline snapshot, extract commit `c34b0ba` into a temporary directory. Use `--no-userinit --no-sysinit` and explicitly initialize ASDF's source registry with the snapshot, trivial-simd and dependency trees, excluding inherited configuration. Set output translations to a writable temporary cache. Load the benchmark scripts from that snapshot and verify `asdf:system-source-directory` points there before measuring.[^registry]

Run three baseline/protocol pairs serially, without other benchmarks or test compilation running concurrently. Matmul uses native BLAS threshold one. See [elementwise measurements](elementwise-performance.md) and [matmul measurements](matmul-performance.md) for workload geometry and timing loops.

## Limitations

- Full-result extension staging and custom-storage input packing are not measured here. Their workspace costs are tracked in [#32](https://todo.sr.ht/~takeiteasy/cl-tensor/32).

[^measurement]: These are local measurements, not cross-platform performance guarantees. Short addition's relative overhead is about 10.7%, with an absolute increase of 0.194 µs. Allocation counters are approximate; the matmul harness samples at most 100 calls, so allocation-region granularity affects per-call figures.
[^registry]: Loading a snapshot ASD alone can still let startup configuration select the working checkout. Isolation and source-directory verification prevent accidentally measuring the same implementation twice.
