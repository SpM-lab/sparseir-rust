# Repository Rules

These are sparse-ir-rs-specific rules. Apply them in addition to the shared
tensor4all rules from `tensor4all-agent-rules`.

The workspace publishes two Rust crates: the core `sparse-ir` crate and the
`sparse-ir-capi` C ABI crate. It also maintains thin Python and Fortran bindings
to that C ABI. Rules below protect the numerical contracts and the boundaries
between those layers.

Existing code that predates these rules is a migration target, not precedent
for new code. Fix violations in the scope of a task when practical, but do not
turn an unrelated change into an unbounded cleanup.

## Source Of Truth And Layering

- The core Rust implementation owns numerical algorithms, basis and sampling
  semantics, working-precision policy, and backend-neutral validation.
- `sparse-ir-capi` owns ABI translation: opaque handles, raw-pointer checks,
  memory-order conversion, status-code mapping, and panic containment. Do not
  move core numerical logic into the C API to avoid adding an appropriate Rust
  abstraction.
- Python and Fortran code in this repository are thin bindings. Do not duplicate
  numerical algorithms there when the Rust core or C ABI owns the behavior.
- Public Rust docs, the checked-in C headers, generated binding declarations,
  and examples must describe the currently implemented surface. Historical
  notebooks and implementation notes are not public API specifications.
- Fix behavior at the layer that owns it. Avoid compatibility shims, duplicated
  validation, and downstream reach-through into private representations.

## Public Surface And Compatibility

- Keep the public Rust API and C ABI deliberate and small. Planning helpers,
  intermediate matrices, scratch buffers, backend dispatch details, and test
  helpers should be private or crate-private unless downstream users are
  expected to rely on them.
- A public Rust item, exported C symbol, status code, opaque-handle lifecycle,
  struct layout, constant value, or calling convention is a compatibility
  contract. Review its Semantic Versioning impact before changing it.
- Do not expose low-level Rust internals through the C ABI merely to satisfy one
  wrapper. Add the narrowest language-neutral operation that preserves the
  core abstraction.
- When a public surface changes, update Rust documentation, C headers, generated
  declarations, wrappers, examples, and integration tests in the same change.
- Existing public items that violate current policy are migration targets, not
  examples to copy into new modules.

## Numerical Correctness And Precision

- Numerical behavior is part of the public contract. Changes to kernels,
  interpolation, Gauss rules, SVE, TSVD, DLR, basis construction, sampling, or
  fitters require tests that check values, reconstruction, or residuals rather
  than only shapes or successful execution.
- Cover representative values and difficult regimes: small and large
  `lambda`, requested `epsilon` near working-precision transitions, interval
  endpoints, roots and extrema, bosonic and fermionic statistics, positive-only
  and full Matsubara grids, real and complex data, and rank-deficient or
  ill-conditioned matrices where applicable.
- Treat `f64` and `Df64` as distinct working-precision contracts. Do not claim
  double-double accuracy for a path whose elementary functions, conversions,
  decomposition backend, or final storage reduce it to `f64` accuracy.
- Keep the automatic `TworkType` selection and epsilon clamping policy
  consistent between the core Rust API and C API constants. A threshold change
  is a numerical behavior change and requires focused boundary tests.
- Coordinate normalization and denormalization must be applied consistently in
  coefficient construction and evaluation. Polynomial and quadrature code must
  preserve the documented interval and endpoint conventions.
- Preserve symmetry, parity, statistics, Matsubara-frequency, singular-value
  ordering, and truncation semantics. Optimizations must not silently change
  these mathematical conventions.
- When parity with SparseIR.jl, Python sparse-ir, or libsparseir is intended,
  state which behavior is being matched and test it with reproducible reference
  data. Distinguish compatibility requirements from independent oracle checks.

## Numerical Test Tolerances

- Choose tolerances before evaluating a candidate change, based on an error
  model, conditioning, working precision, a higher-precision oracle, or a
  documented cross-language compatibility bound.
- Do not relax a tolerance merely until a failing result passes. A relaxation
  must record the observed residual, the previous and proposed bounds, the
  affected parameter range, and why the new bound still detects meaningful
  regressions.
- Prefer high-precision references such as `DBig`, analytic values, known
  identities, reconstruction residuals, or independently generated
  SparseIR.jl/Python reference data.
- Approximate comparisons should report a useful diagnostic such as maximum
  absolute error, relative norm error, orthogonality residual, or reconstruction
  residual.
- Keep reference-data generation reproducible. Record the upstream project and
  version or commit, generator path, parameters, precision, and serialization
  convention near the fixture or generator.

## Public Boundary Validation And Errors

- Validate user-derived `lambda`, `epsilon`, statistics, rank, dimensions,
  target axes, sampling points, intervals, matrix shapes, and configuration
  values before allocation, indexing, backend calls, or no-op shortcuts.
- Public Rust library paths must not turn invalid user input into `panic!`,
  `unwrap`, `expect`, unchecked indexing, or debug-only assertions. Prefer
  crate-local typed errors with enough context for C and language bindings to
  report the failure.
- Panics are reserved for genuinely internal invariants whose proof is local.
  Existing public panics are migration targets and must not be copied into new
  APIs.
- Preserve error categories across layers. Known invalid input must map to a
  specific `SPIR_INVALID_*` or dimension status, unsupported operations to
  `SPIR_NOT_SUPPORTED`, and unexpected internal failures to
  `SPIR_INTERNAL_ERROR`.
- Do not collapse an available typed Rust error into an internal error merely
  because the C ABI currently carries only a status code. Keep the mapping
  centralized and testable.

## C ABI Safety

- No Rust panic may unwind across an `extern "C"` boundary. Every exported
  entry point that can panic must use the repository's panic-containment pattern
  and return the documented failure status or null result.
- `catch_unwind` does not make an invalid non-null pointer safe. The caller owns
  the validity and lifetime contract for non-null pointers; Rust must still
  check nullability, lengths, dimensions, enum values, and other verifiable
  preconditions before dereferencing.
- Before constructing a slice or tensor view from raw parts, validate pointer
  requirements and use checked arithmetic for dimension products, element
  counts, byte lengths, strides, and offsets.
- Check conversions among C integer types, `usize`, BLAS LP64 `i32`, and BLAS
  ILP64 `i64`. Reject values that are negative or out of range before casting.
- Validate output capacity, shape, and element type before writing. Unless an
  API explicitly documents partial output on failure, validation must complete
  before mutating caller-owned output buffers.
- Do not accept overlapping input and output buffers unless the operation's
  aliasing contract explicitly permits it and the implementation is tested for
  that overlap.
- Unsafe helper functions must state their complete caller obligations in a
  `# Safety` section; the call site must establish those obligations locally.

## Opaque Handle Ownership

- Every C opaque handle has one documented constructor/clone/release ownership
  model. A successful constructor returns an owned handle, clone returns a new
  independently releasable handle, and release consumes exactly one handle.
- Releasing a null pointer may be a no-op when documented. Double release,
  dereference after release, or passing an arbitrary non-null pointer remains a
  caller contract violation and is not recoverable with `catch_unwind`.
- Keep Rust implementation details behind opaque handles. Do not expose inner
  enum layouts, `Arc` internals, Rust vtables, or allocator-specific ownership
  through the ABI.
- Sharing an `Arc` proves shared lifetime, not mutable access. Do not derive
  mutable aliases from cloneable handles without a separate exclusivity
  contract.
- Constructors with an out-status parameter must set it deterministically on
  every success and failure path. A null return and status value must not
  contradict each other.

## Memory Order And Dimension Semantics

- The C API supports both row-major and column-major buffers. Each function must
  document the logical shape, memory order, and meaning of every target or batch
  dimension.
- Keep row-major/column-major translation in shared C API helpers. Do not add
  operation-local dimension reversal or axis remapping when the common
  conversion path can express it.
- A memory-order conversion must preserve the logical tensor. Add tests that
  feed the same non-square, rank-greater-than-two value in both orders and
  compare logical results.
- Do not hide layout copies in core numerical APIs. When an ABI or backend
  requires packing or permutation, keep the copy at an explicit boundary and
  document its cost.
- Test singleton, zero-length where supported, non-square, batched, and
  target-dimension edge cases. Shape products and axis remapping must be checked
  before creating mdarray views.

## BLAS And GEMM Backend Contract

- The pure-Rust/faer backend, compile-time `system-blas` backend, and runtime
  injected BLAS function pointers must implement the same mathematical and
  memory-order contract.
- LP64 and ILP64 are distinct ABI choices. Keep function-pointer types,
  dimensions, leading dimensions, and tests separate; never infer one ABI from
  pointer presence alone.
- Validate all matrix dimensions and leading dimensions before an FFI call.
  Overflow or an unsupported dimension must return an error rather than narrow
  silently.
- Backend fallback must be intentional. If a fallback allocates, packs, changes
  threading, or uses a slower implementation, make that boundary visible in the
  implementation and cover it with correctness and performance tests.
- Do not select a backend independently inside inner numerical helpers. Route
  GEMM through the workspace's established dispatcher or explicit backend
  handle so one operation does not mix incompatible providers.
- Benchmarks comparing providers must use release builds, pin relevant thread
  counts, separate setup from execution, and report the BLAS implementation and
  LP64/ILP64 mode.

## Unsafe Code And Working Buffers

- Keep `unsafe` localized to FFI, backend adapters, raw buffer management, and
  view construction that cannot be expressed safely. Numerical orchestration
  and public validation should remain safe Rust.
- Every unsafe block requires a nearby `// SAFETY:` comment explaining pointer
  validity, alignment, initialized range, bounds, lifetime, and aliasing facts
  relevant to that block.
- `WorkingBuffer` and similar reusable scratch storage must maintain checked
  byte capacity and alignment before producing typed slices. A typed mutable
  slice must never outlive or alias another mutable view of the buffer.
- Distinguish uninitialized full-overwrite storage from initialized
  read-before-write storage. Do not fix an initialization bug by unconditionally
  zero-filling a shared hot-path buffer.
- Unsafe `Send` or `Sync` implementations require a concrete ownership and
  aliasing argument plus tests that exercise movement or sharing across threads
  when that behavior is supported.

## Generated Bindings And Cross-Language Synchronization

- Treat `sparse-ir-capi/assets/sparse_ir_capi.h` and
  `sparse-ir-capi/include/sparseir/sparseir.h` as checked-in ABI surfaces. Keep
  them synchronized with exported Rust symbols and constants.
- `python/pylibsparseir/ctypes_autogen.py` is generated by
  `python/tools/gen_ctypes.py`. Regenerate it; do not hand-edit generated
  declarations to conceal drift.
- Files under `fortran/src/` marked as generated must be changed through their
  scripts under `fortran/script/`. Keep generated declarations, public exports,
  and implementation include files synchronized.
- A C ABI change is incomplete until the relevant C/C++ integration tests,
  Python tests, and Fortran tests compile and exercise the changed contract.
- Keep complex-number layout, boolean representation, enum/status constants,
  pointer ownership, and row/column-major conventions identical across Rust,
  headers, ctypes, C++, and Fortran declarations.
- Generated-file checks should be reproducible and fail CI when regeneration
  would produce a diff.

## Performance And Allocation Discipline

- Avoid heap allocation, whole-array cloning, formatting, and shape decoding in
  element-sized hot loops. Reuse working buffers, precompute interpolation
  coefficients, and carry prepared shape/stride metadata into inner kernels.
- Prefer borrowed mdarray views and explicit output buffers over temporary dense
  tensors. Do not zero-initialize outputs that the operation fully overwrites.
- Cell and segment lookup should retain logarithmic or better lookup behavior
  where the grid permits it. Do not replace binary search with an unbounded
  linear scan without measured justification.
- Performance changes require representative sizes, parameter regimes, memory
  orders, working precisions, and providers. A single favorable matrix size is
  not sufficient evidence.
- A performance optimization must preserve the numerical and ABI contracts in
  this file. Do not trade away validation, precision, or deterministic ownership
  for an unmeasured speedup.

## Tests And Documentation

- Tests follow implementation ownership: core numerical behavior belongs in
  `sparse-ir`, ABI behavior in `sparse-ir-capi`, and wrapper marshaling in the
  corresponding language binding.
- Rust examples and public documentation should assert known values,
  reconstruction accuracy, or meaningful invariants. Examples that only print
  results or check shapes are insufficient for numerical workflows.
- Public `Result` APIs should document concrete error conditions. Exported C
  functions should document nullability, ownership, input/output dimensions,
  memory order, status results, and panic containment.
- When a public API or supported capability changes, audit the workspace
  README, crate READMEs, rustdoc, C headers, Python and Fortran documentation,
  examples, and ecosystem links for stale claims.
- Keep tests deterministic. Record random seeds, reference-generation inputs,
  provider configuration, and thread counts when they affect results.
- Local checks should be proportional to the change. Hosted CI owns the full
  Rust, C/C++, Python, Fortran, feature, and provider matrix; changes to a
  boundary must still run the focused checks for that boundary before review.

## Provenance And Scientific Credit

- When implementing or translating code while reading SparseIR.jl, Python
  sparse-ir, libsparseir, nalgebra, or another project, record the project,
  source file or symbol, revision when practical, and whether the code was
  ported, derived, convention-matched, or only validated against it.
- A close translation retains upstream copyright and license obligations.
  Preserve notices and license texts; do not describe derived code as an
  independent implementation.
- Cite the relevant SparseIR and numerical-algorithm literature in the
  repository's established citation surface. Scientific credit is required
  even when an algorithm is implemented independently and copyright does not
  apply.
- Reference fixtures must identify their generator and upstream version. Do not
  commit unexplained numerical tables whose provenance cannot be reproduced.

## File Organization

- Keep files focused by responsibility, but do not split solely to satisfy a
  line-count target. Large core or C API files are review triggers, not
  automatic violations.
- Prefer boundaries such as validation, numerical preparation, execution,
  backend dispatch, ABI translation, handle lifecycle, and generated bindings.
  Avoid arbitrary `part1`/`part2` modules.
- Keep production logic separate from large test suites. Existing module-local
  test files may remain; new tests should follow the ownership and organization
  already used by the affected module.
- Do not encode a current source-tree listing as a durable rule. Module
  organization may evolve as long as ownership and public boundaries stay
  clear.
