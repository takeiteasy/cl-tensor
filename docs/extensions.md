# Dtype and storage extensions

External packages register dtype descriptors and implement CLOS methods for storage and operations. Tensor APIs use keyword dtype names. Packed formats keep their layout and kernels in the extension.

## Register a dtype

```lisp
(defclass packed8 (ct:dtype) ())
(ct:register-dtype
 (make-instance 'packed8 :name :my-packed8 :element-type 'single-float
                        :block-size 32 :block-bytes 34 :storage-only t))
```

| Descriptor initarg | Meaning |
|---|---|
| `:name` | Unique keyword, read with `dtype-name` |
| `:element-type` | Logical scalar Lisp type; built-in encoded floats retain their raw-bit type |
| `:block-size` | Positive logical element count per block; default 1 |
| `:block-bytes` | Positive stored bytes per block |
| `:zero` | Default scalar zero for ordinary allocation; default 0 |
| `:storage-only` | Disables ordinary computational fallback; default NIL |
| `:operations` | Optional property list from operation keywords to selector functions |

`find-dtype` returns the registered descriptor. `dtype-block-size` and `dtype-block-bytes` expose its block geometry. `dtype-bytes` returns bytes per logical element, including fractions such as `17/16`; it is not an address calculation for packed storage.

Registration rejects duplicate names. Registering the identical object again succeeds. Registered metadata and `*dtypes*` are read-only by convention; use registration to extend the registry. Register during initialization, before concurrent tensor operations.[^registry]

## Implement storage

| Generic method | Contract |
|---|---|
| `storage-length (storage)` | Logical element capacity |
| `storage-element-type (storage)` | Logical scalar Lisp type |
| `storage-dtype (storage)` | Inferred dtype keyword; override for an encoded storage kind |
| `storage-readable-p (storage)` | Whether logical scalar reads are available; default NIL |
| `storage-writable-p (storage)` | Whether logical scalar writes are available; default NIL |
| `storage-ref (storage index)` | Read one logical scalar |
| `(setf storage-ref) (value storage index)` | Write one logical scalar |

Simple vectors and trivial-simd vector views implement these methods. Missing scalar access signals `unsupported-storage-access`. Packed storage may omit both access methods and rely on operation and copy methods instead.

Offsets and strides count logical elements. Core validates dimensions, stride rank, offsets and reachable bounds; then it calls `validate-storage-view`. Every metadata view goes through this validation. Extensions can reject partial blocks, transposes, broadcasts or other layouts they cannot represent.

## Implement dtype methods

These methods receive a registered descriptor as their first argument.

| Generic method | Contract |
|---|---|
| `allocate-storage (dtype shape)` | Independent zeroed storage for the requested shape |
| `coerce-scalar (dtype value)` | Validate/coerce a scalar constructor or assignment value |
| `validate-storage-view (dtype storage shape strides offset)` | Validate storage compatibility and extension layout restrictions; signal on failure |
| `copy-storage-supported-p (dtype out)` | Whether destination copying is supported |
| `copy-storage! (dtype out input)` | Copy equal-shape, equal-dtype tensors; return `out` |

Defaults allocate ordinary one-element-block arrays, enforce dtype/storage element-type compatibility, and copy through logical scalar access. Packed dtypes supply allocation and copying explicitly. Scalar constructors require writable storage; storage-only descriptors reject ordinary filled constructors. `zeros` uses the allocation method.

`copy-tensor`, copying reshape paths and concatenation use the copy protocol. A packed extension can support these without scalar access, provided its view validator accepts the required layouts. Copy methods receive independent source/destination storage from core; callers of the public copy method must handle aliases themselves or supply disjoint storage.

## Select an operation

Specialize `resolve-operation (dtype operation inputs options)`. Return two values: the result dtype keyword and an executor function. Return NIL to decline. Selection validates accepted dtype combinations and extension-specific options, without writing storage.

```lisp
(defmethod ct:resolve-operation ((dtype packed8) operation inputs options)
  (when (and (eq operation :matmul)
             (eq (ct:tensor-dtype (first inputs)) :my-packed8)
             (eq (ct:tensor-dtype (second inputs)) :f32))
    (values :f32 #'packed-matmul!)))
```

The executor has signature `(out inputs options)`. It fills all logical result elements, reads inputs without modifying them, and respects their validated layouts. Its return value is ignored. The core allocates an independent result and checks a supplied destination's shape, dtype, uniqueness and copy support before invoking the executor. Destination-writing calls copy the completed result into the caller's destination, preserving aliased inputs.[^staging]

Core computes broadcasting, matmul and reduction shapes. Axes in hook options are normalized, sorted lists of nonnegative axis numbers; an empty list means no axes. Extensions choose the result dtype, including mixed matmul such as packed weights × `:f32` activations → `:f32`.

| Operation keys | Inputs and options |
|---|---|
| Elementwise public names as keywords | Operands in public argument order; `:dtype`, `:operator` |
| `:sum`, `:mean`, `:prod`, `:minimum`, `:maximum`, `:argmin`, `:argmax` | One tensor; `:axis`, `:keepdims`, `:accumulate` |
| `:softmax`, `:rmsnorm` | Input, then optional RMSNorm weights; `:axis`, `:accumulate`, `:epsilon` |
| `:matmul`, `:dot` | Left and right tensors; no options |
| `:convert` | Input tensor; requested `:dtype`, `:rounding` |
| `:dequantize` | Input tensor; requested `:dtype` |

Selection visits distinct input descriptors in operand order, then the requested destination dtype for conversion or an explicit dtype option. The first accepting method wins. Built-in methods decline unless their operation table supplies a selector. An operation-table selector receives `(dtype operation inputs options)` and follows the same two-value contract.

Vector–vector matmul tries `:dot`, then `:matmul`. Conversion hooks must return the requested dtype. `(dequantize tensor &key (dtype :f32))` tries `:dequantize`, then ordinary `astype` dispatch. Both return independent storage.

## Fallback and example

When all selectors decline, built-in dtypes use existing kernels. Readable custom storage is packed into ordinary arrays; destination writes use independent results and protocol copying. Custom dtypes without accepting methods signal `unsupported-operation`; computation does not implicitly dequantize.

Ordinary arrays retain built-in dtype inference even after extensions register scalar types. Encoded/custom storage overrides `storage-dtype` or supplies `:dtype` explicitly.

[The runnable example](../examples/extensions.lisp) defines an opaque format with two signed-byte codes and one float scale per block. Its Lisp demonstration kernels decode before computing; production extensions supply their own kernels without changing core.

```lisp
(load "examples/extensions.lisp")
(cl-tensor/extension-example:run-example)
;; => :F32 vector (17 39), and independent decoded weights
```

See [dispatch measurements](extensions-performance.md) for complete-call time and allocation with built-in storage.

## Limitations

- Extension out-forms stage a complete result; accessible-storage fallback packs complete inputs. Bounded staging and alias-safe direct execution are tracked in [#32](https://todo.sr.ht/~takeiteasy/cl-tensor/32).
- Production quantized formats and model integration belong to the inference project: [#24](https://todo.sr.ht/~takeiteasy/cl-tensor/24).

[^registry]: Descriptor replacement requires a fresh Lisp image. Reloading methods on an existing descriptor class is supported; duplicate registration does not replace its metadata. Concurrent registry mutation is outside the protocol.
[^staging]: Executor failures leave a supplied destination unchanged. A failing final copy method can partially update it; extensions validate their copy prerequisites before writing.
