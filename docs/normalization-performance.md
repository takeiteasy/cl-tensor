# Normalization measurements

Softmax and RMSNorm use typed Lisp rows. Softmax obtains maxima through trivial-simd; RMSNorm applies weights through its N-D multiplication, and out-forms use same-dtype N-D copying. The comparison kernels remain in the benchmark script.

## Workloads

The benchmark measures complete allocating and destination-writing calls, including tensor validation, group packing, weights and output copying. Apple M1, SBCL 2.6.8, Lisp and native backends, 2026-10-07.

| Dimension | Cases |
|---|---|
| Dtype | f32, f64 |
| Accumulation | Input precision, f64 sums |
| Shape | 1 or 32 rows; widths 32, 128, 512, 4,096 and 32,768 |
| Layout | Contiguous, transposed, reversed inner axis, contiguous foreign storage |
| API | Allocating and out-forms of softmax and RMSNorm |
| RMSNorm weights | Scalar 1.5 |

There are 1,280 workloads, each measured with typed and kernel-assisted row execution. Input values repeat quarter steps in [-2, 2]. Each path is checked against an independent double-precision scalar reference before timing, using `1e-5 × (1 + |reference|)` for f32 and `1e-12 × (1 + |reference|)` for f64.

## Compared paths

| Operation/mode | Kernel-assisted execution |
|---|---|
| Softmax, input accumulation | Nested maximum, exponential sum and output kernel |
| Softmax, widened f32 sums | Shifted-exponential kernel, typed double sum and single-float scaling |
| RMSNorm, input accumulation | Sum-of-squares kernel, typed scaling |
| RMSNorm, widened f32 sums | Square kernel, typed double sum and single-float scaling |

Both alternatives use the same tensor layout, weights and destination handling. The nested softmax kernel computes exponentials twice. Widened f32 alternatives preserve single-float exponentials and squares before double accumulation.

## Results

Medians across three fresh processes retain typed rows for both operations. All 7,680 measured paths pass their numerical checks. Ratios below compare kernel latency with typed latency; values above one mean the kernel is slower.

| Operation | Workloads | Geometric-mean ratio | Worst ratio | Production path |
|---|---:|---:|---:|---|
| Softmax | 640 | 1.439 | 4.575 | Typed |
| RMSNorm | 640 | 0.943 | 2.486 | Typed |

Kernel softmax is 44% slower overall. Kernel-assisted RMSNorm is 6% faster overall, but its worst regression is 2.49×. Neither meets the 10% overall improvement and 10% maximum regression thresholds.

Representative native f32 out-forms with input-precision accumulation, contiguous inputs and preallocated destinations:

| Operation | Rows × width | Typed µs | Kernel µs | Typed bytes | Kernel bytes |
|---|---:|---:|---:|---:|---:|
| Softmax | 1 × 32 | 4.73 | 7.43 | 2,621 | 4,587 |
| Softmax | 1 × 4,096 | 52.87 | 45.24 | 19,324 | 20,825 |
| Softmax | 32 × 512 | 215.66 | 265.25 | 74,026 | 142,843 |
| RMSNorm | 1 × 32 | 6.26 | 6.37 | 3,931 | 4,586 |
| RMSNorm | 1 × 4,096 | 45.38 | 34.19 | 20,629 | 20,676 |
| RMSNorm | 32 × 512 | 167.03 | 128.19 | 67,934 | 72,015 |

Raw timings and allocations: [run 1](benchmark-runs/2026-10-07-normalization-1.txt), [run 2](benchmark-runs/2026-10-07-normalization-2.txt), [run 3](benchmark-runs/2026-10-07-normalization-3.txt).

## Reproduce

```sh
sbcl --non-interactive --load tests/normalization-bench.lisp
```

Use a fresh process for each of three runs. Timings report the median of three batches after warmup and calibration to at least 5 ms per batch. Allocation sampling uses up to 100 additional calls after a full GC.[^measurement]

Selection uses each workload's median latency across the three processes. A kernel strategy qualifies only when its geometric-mean latency is at least 10% lower and no workload is more than 10% slower. Selection applies once per operation across both backends, dtypes, accumulation modes and layouts; production has no width crossover dispatch.

## Limitations

- Measurements describe this M1 and these workloads, not other platforms or whole-model inference.
- Normalization uses O(input elements) result scratch. Portable f64 weight/output stages allocate per element, foreign extrema use portable staging, and restoring original axis order can require another tensor: [#8](https://github.com/communal-software/cl-tensor/issues/8).
- Irregular groups use the existing reduction packing helper: [#7](https://github.com/communal-software/cl-tensor/issues/7).
- Native dependent-pass kernels pay per-row/pass setup costs: trivial-simd #137.

[^measurement]: The script temporarily replaces the private row dispatcher inside the benchmark process and restores it after each case, including on errors. `CT_NORMALIZATION_CHECK_ONLY=1` loads candidates without running timings. SBCL's allocation counter has sampling granularity and excludes native allocations. Both candidates retain their ordinary scalar system math and backend-dependent exceptional-value behavior.
