# Extension staging measurements

Bounded input packing reduces allocation by 42–45% and complete-call time by 14–21% for the large f32 custom-storage cases. Safe direct opaque negation removes full-result staging and runs about 15–16× faster at 65,536 elements.[^local]

## Custom storage and executors

Apple M1, SBCL 2.6.8, native backend, 2026-10-07. Baseline is `8e773b1`; results are medians of three serial fresh-process trials. Time includes selection, validation, computation and final copying where applicable. Inputs and destinations are preallocated.

| Elements and case | Baseline µs/call | Implementation µs/call | Baseline bytes/call | Implementation bytes/call |
|---|---:|---:|---:|---:|
| 1024 wrapped-contiguous | 148.141 | 124.621 | 10,433 | 10,689 |
| 1024 opaque-disjoint | 31.987 | 2.418 | 3,514 | 262 |
| 65536 wrapped-contiguous | 9,634.250 | 8,030.500 | 524,320 | 303,102 |
| 65536 wrapped-reverse | 9,787.625 | 8,077.750 | 524,320 | 303,102 |
| 65536 wrapped-stride2 | 9,695.375 | 8,066.875 | 524,320 | 303,102 |
| 65536 wrapped-broadcast | 9,710.000 | 7,715.375 | 524,320 | 303,102 |
| 65536 wrapped-overlap | 10,338.125 | 8,881.875 | 524,320 | 303,102 |
| 65536 opaque-disjoint | 1,964.781 | 125.551 | 192,859 | 256 |
| 65536 opaque-exact-alias | 1,915.781 | 123.197 | 192,859 | 256 |
| 65536 opaque-declined-shift | 1,906.969 | 1,981.562 | 192,859 | 192,859 |
| 262144 wrapped-contiguous | 37,517.500 | 32,080.500 | 2,097,184 | 1,146,888 |
| 262144 opaque-disjoint | 7,585.250 | 519.813 | 786,464 | 0 |

Wrapped cases add scalar zero to readable f32 custom storage. They cover ordinary, reversed and stride-two destinations, zero-stride inputs and exact destination/input aliases. Opaque cases negate paired signed-byte codes with per-pair scales; separate buffers and exact aliases select direct execution. Shifted aliases decline and retain full-result staging. The small 1,024-element fallback call adds about 256 bytes of metadata even though its input buffer is no smaller than the input.

## Built-in regression controls

The combined trials show some 3–5% variation in built-in controls. Three additional fresh-process trials run only the existing elementwise and matmul benchmarks, with implementation first and baseline second. Their median times show no repeatable slowdown; allocation matches the baseline in every control case.

| Control | Baseline µs/call | Implementation µs/call | Bytes/call, both |
|---|---:|---:|---:|
| Native add, 1,024 contiguous f32 | 2.014 | 2.020 | 1,887 |
| Native add, 65,536 contiguous f32 | 10.245 | 10.266 | 1,867 |
| Native add, 65,536 transposed f32 | 87.401 | 87.334 | 1,933 |
| Lisp matmul, interleaved f64 matrices | 137.104 | 136.279 | 141,449 |
| Native matmul, interleaved f32 matrices | 133.715 | 131.055 | 141,453 |
| Native matmul, contiguous f64 batches | 23.112 | 23.094 | 3,276 |

Retained control [baseline 1](benchmark-runs/2026-10-07-staging-control-before-1.txt), [2](benchmark-runs/2026-10-07-staging-control-before-2.txt), [3](benchmark-runs/2026-10-07-staging-control-before-3.txt); [implementation 1](benchmark-runs/2026-10-07-staging-control-after-1.txt), [2](benchmark-runs/2026-10-07-staging-control-after-2.txt), [3](benchmark-runs/2026-10-07-staging-control-after-3.txt).

## Workspace and failure checks

Core allocates at most 4,096 logical input elements per tensor operand for elementwise fallback, plus one full independent result. Allocation-method instrumentation checks empty, scalar, exact-boundary and partial-final chunks. Accepted opaque direct calls allocate no core result tensor; declined selectors allocate a full result.

Tests cover reversed/transposed destinations, shifted and hidden aliases, masks, integer/float/complex dtypes, invalid destinations rejected before reads or execution, and unchanged destinations after staged executor and late input-read failures. SBCL additionally injects a second-chunk kernel failure. Opaque direct negation preflights all codes before writing and rejects malformed storage geometry.

The data-buffer bound describes live input workspace, not total bytes allocated. Kernel/view metadata and generic scalar access can allocate; the full-result buffer remains proportional to output size.[^counter]

## Reproduce

```sh
sbcl --non-interactive --load tests/extensions-bench.lisp
```

The harness warms each case, doubles iterations until a timing batch lasts at least 50 ms, reports the median of three timed batches, and samples allocation over at most 1,000 calls. It prints the system source directory and backend.

For baseline runs, extract `8e773b1` into a temporary directory. Use the current benchmark harness for both implementations. Start SBCL with `--no-userinit --no-sysinit`, explicitly initialize ASDF's source registry with the selected checkout, trivial-simd and dependencies, exclude inherited configuration, and translate compilation output into a temporary cache. Assert that `asdf:system-source-directory` resolves to the selected checkout before loading the harness.

Each retained trial runs the extension harness followed by the existing elementwise and matmul benchmarks. For the additional controls, omit the extension harness and reverse each pair's order. Run trials serially without concurrent tests or compilation. Retained [baseline 1](benchmark-runs/2026-10-07-staging-before-1.txt), [2](benchmark-runs/2026-10-07-staging-before-2.txt), [3](benchmark-runs/2026-10-07-staging-before-3.txt); [implementation 1](benchmark-runs/2026-10-07-staging-after-1.txt), [2](benchmark-runs/2026-10-07-staging-after-2.txt), [3](benchmark-runs/2026-10-07-staging-after-3.txt).

## Limitations

- Staged extension calls and elementwise fallback retain full-result scratch; reductions, normalization, matmul and conversion fallback pack complete custom-storage inputs: [#33](https://todo.sr.ht/~takeiteasy/cl-tensor/33).
- Measurements cover f32 scalar-access storage and the opaque demonstration format. They do not establish workspace or timing guarantees for third-party executors.

[^local]: Local measurements describe this machine and workload, not cross-platform performance guarantees. Direct execution depends on the extension's declared alias and failure guarantees.
[^counter]: SBCL allocation-region refill granularity affects bytes/call. A reported zero does not mean no Lisp objects are allocated; allocation-method instrumentation establishes the absence of core result tensors independently.
