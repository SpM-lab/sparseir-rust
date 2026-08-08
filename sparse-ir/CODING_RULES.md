# sparse-ir Implementation Notes

This file contains supplemental implementation notes for the `sparse-ir`
crate. It is not the repository policy source of truth.

Before changing code, read the root [`AGENTS.md`](../AGENTS.md), the shared
tensor4all rules it references, and the root
[`REPOSITORY_RULES.md`](../REPOSITORY_RULES.md). When this file conflicts with
those sources, follow the repository and shared rules.

Do not treat an existing implementation pattern as automatically correct or
use this document as authority for a broad cleanup. Non-local migrations require
the scope and approval described in `REPOSITORY_RULES.md`.

## Numeric Types And Working Precision

- The current extended-precision type is `xprec::Df64`, re-exported by
  `sparse-ir`. Do not introduce new public APIs using the historical
  `TwoFloat` name.
- Generic numerical helpers should support `f64` and `Df64` through the
  established traits when both working precisions need the same semantics.
  Keep dtype dispatch at an outer boundary rather than duplicating complete
  algorithms.
- `Df64` storage does not by itself guarantee double-double accuracy. Elementary
  functions, conversions, matrix operations, decomposition backends, and final
  `f64` outputs may limit the attainable precision. Document the limiting step
  when it affects an algorithm or public accuracy claim.
- Working-precision selection and epsilon clamping belong to the SVE precision
  policy. Do not add operation-local thresholds that bypass `TworkType` and its
  shared helpers.
- Convert constants explicitly through the numeric abstraction used by the
  algorithm. Avoid unnecessary `f64` round trips in a `Df64` path.

## Interpolation And Quadrature

- Normalize interpolation coordinates to `[-1, 1]` for Legendre evaluation and
  apply the inverse convention consistently during evaluation.
- Obtain segmentation, polynomial degree, and Gauss-point configuration from
  the owning hints or strategy objects. Do not duplicate their thresholds as
  magic numbers in downstream helpers.
- Precompute reusable interpolation coefficients and quadrature metadata during
  construction rather than rebuilding them in each evaluation.
- Preserve logarithmic segment or cell lookup where the grid permits binary
  search. A different traversal requires correctness tests and performance
  evidence for representative grids.

## Errors And Invariants

- Reachable invalid public input should return the typed error required by
  `REPOSITORY_RULES.md`; it is not an unrecoverable condition merely because an
  older implementation panics.
- Keep a panic or assertion only for a genuinely internal invariant whose proof
  is local. Do not copy an existing panic into a new public API.
- C ABI status mapping, panic containment, raw-pointer validation, and opaque
  handle ownership belong to `sparse-ir-capi`, not this core crate.
- Explain non-obvious numerical invariants and precision limitations near the
  implementation. Comments should state why the invariant holds or which
  operation limits precision, not merely restate the code.

## Numerical Tests

- Use analytic values, reconstruction identities, independently generated
  SparseIR reference data, or higher-precision values such as `DBig` as
  appropriate for the algorithm.
- Exercise both `f64` and `Df64` when a generic path claims to support both, and
  include difficult parameter regimes identified in `REPOSITORY_RULES.md`.
- Choose and justify tolerances according to the repository tolerance policy.
  Do not use the historical fixed tolerances from earlier versions of this
  document as universal defaults.
- Report useful diagnostics such as maximum absolute error, relative norm,
  orthogonality error, or reconstruction residual.
- Keep tests with their implementation owner. Module-local test files under
  `src/` and public integration tests under `tests/` are both valid; choose the
  location based on whether the behavior requires private or public access.
- Development-only debug output should not remain in committed tests unless it
  is a stable failure diagnostic.

## Provenance

- When following or translating SparseIR.jl, Python sparse-ir, libsparseir,
  nalgebra, or another implementation, follow the provenance, copyright,
  scientific-credit, and user-approval requirements loaded through
  `AGENTS.md`.
- Reference comparisons should identify the upstream project and version or
  commit, generator, parameters, and precision so later contributors can
  reproduce them.
