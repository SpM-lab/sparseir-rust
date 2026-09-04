# Repository Rules

These are sparse-ir-rs-specific facts and conventions. Apply them in addition
to the shared [SpM-lab agent rules](https://github.com/SpM-lab/spm-agent-rules).
Where the two disagree, follow this file and note the override in the pull
request.

## Workspace Layout

This is a Cargo workspace (`resolver = "2"`) with two published members,
defined in the top-level `Cargo.toml`:

- **`sparse-ir/`** — the core Rust crate (`sparse_ir`, lib crate types
  `cdylib` + `rlib`). Implements the sparse IR algorithms: kernels and SVE
  (`kernel.rs`, `kernelmatrix.rs`, `tsvd.rs`), piecewise Legendre polynomials
  and their Fourier transforms (`poly.rs`, `polyfourier.rs`), 1D/2D
  interpolation (`interpolation1d.rs`, `interpolation2d.rs`), Gauss
  quadrature (`gauss.rs`), the `FiniteTempBasis`/DLR basis types
  (`basis.rs`, `basis_trait.rs`, `dlr.rs`), tau/Matsubara sampling
  (`sampling.rs`, `matsubara_sampling.rs`, `taufuncs.rs`, `freq.rs`),
  special functions (`special_functions.rs`), the GEMM dispatch layer
  (`gemm.rs`, `col_piv_qr.rs`), and numeric/precision plumbing
  (`numeric.rs`, `working_buffer.rs`, `fpu_check.rs`). Test modules live
  next to their implementation as `*_tests.rs` files included via `#[cfg(test)]`
  from the corresponding `src/*.rs` module (e.g. `basis.rs` includes
  `basis_tests.rs`); additional integration tests live in `sparse-ir/tests/`.
  Supplemental, non-authoritative implementation notes for this crate are in
  [`sparse-ir/CODING_RULES.md`](sparse-ir/CODING_RULES.md).
- **`sparse-ir-capi/`** — the C ABI crate (`sparse_ir_capi`, lib crate types
  `cdylib` + `staticlib` + `rlib`). Depends on `sparse-ir` and translates it
  to a stable C interface: opaque handles and status codes (`types.rs`,
  `macros.rs`, `utils.rs`), and per-domain exported functions in
  `basis.rs`, `dlr.rs`, `funcs.rs`, `gemm.rs`, `kernel.rs`, `sampling.rs`,
  `sve.rs`. Integration and regression tests live in `sparse-ir-capi/tests/`
  (`integration_test.rs`, `test_capi_custom_sampling.rs`,
  `test_funcs_deriv.rs`); ad hoc debug scripts used while developing the ABI
  are under `sparse-ir-capi/examples/` (mostly `.jl` scripts, plus
  `test_julia.jl`).

Other top-level directories are thin bindings or consumers of the C API, not
part of the Cargo workspace:

- `python/` — the `pylibsparseir` ctypes wrapper package (see
  `python/tools/gen_ctypes.py` below).
- `julia/` — `SparseIR.jl`-adjacent build tooling (e.g.
  `julia/build_tarballs.jl`) used for downstream version bumps; the Julia
  wrapper package itself lives in the separate `SpM-lab/SparseIR.jl`
  repository.
- `fortran/` — Fortran bindings to the C API (`fortran/src/`, generated via
  scripts in `fortran/script/`).
- `cxx_tests/` — a standalone CMake project that links the built
  `sparse-ir-capi` library and header and exercises them from C++
  (`cinterface_core.cxx`, `cinterface_integration.cxx`).
- `capi_benchmark/` — benchmarking harness against the C API.
- `notebook/`, `docs/` — supporting notebooks and documentation.

## Building And Testing

All commands below were verified to run in this checkout (macOS, Rust 1.96,
Accelerate framework as the default BLAS backend) and mirror what CI runs in
`.github/workflows/`.

```bash
# Build/test the core crate (default backend: faer, pure Rust)
cargo build -p sparse-ir --release
cargo test -p sparse-ir --release

# Build/test the whole workspace
cargo build --all-targets --release --locked
cargo test --all-targets --release --locked

# Build/test sparse-ir with the system-BLAS (LP64) GEMM backend instead of faer
cargo build -p sparse-ir --features system-blas --all-targets --release --locked
cargo test -p sparse-ir --features system-blas --all-targets --release --locked

# Build/test the C API crate
cargo build -p sparse-ir-capi --release
cargo test -p sparse-ir-capi --release

# Run the DLR/IR/sampling round-trip example
cargo run --example roundtrip --release
```

Feature flags defined in `sparse-ir/Cargo.toml`:

- `system-blas` — switches the GEMM backend from the default pure-Rust
  `faer` implementation to system BLAS via `blas-sys` (LP64). Requires
  `libopenblas-dev`/`pkg-config` on Linux CI; macOS uses the Accelerate
  framework automatically (see the `build.rs` warning emitted during
  compilation).
- `shared-lib` — enables shared-library-oriented build settings.

`sparse-ir-capi/Cargo.toml` mirrors `system-blas`, forwarding it to
`sparse-ir/system-blas`, plus its own `capi` feature.

Git hooks (via `cargo-husky`, `dev-dependencies` in `sparse-ir/Cargo.toml`)
run `cargo fmt --all -- --check` on commit
(`.cargo-husky/hooks/pre-commit`); if formatting fails, run
`cargo fmt --all` before committing.

Version consistency across Rust/Python/Julia metadata is checked with:

```bash
python3 check_version.py
```

This reads the canonical version from `[workspace.package]` in `Cargo.toml`.
See the "Version management" section of `README.md` and
`bump_version_downstream.md` for the release process itself; do not duplicate
that process here.

## C API Surface

- `sparse-ir-capi/src/` is the Rust source of the C ABI: exported `spir_*`
  functions, opaque handle types, and status codes.
- `sparse-ir-capi/include/sparseir/sparseir.h` is the canonical C header,
  generated from the Rust exports via `cbindgen` (config:
  `sparse-ir-capi/cbindgen.toml`; excludes internal types such as
  `BasisType`, `FuncsType`, `KernelType`, `SamplingType`, `SVEResultType`,
  `Arc_SVEResult`, and the `inner` field). Do not hand-edit this file — CI's
  `header-sync` job in `.github/workflows/rust.yml` regenerates it with
  `cbindgen --config cbindgen.toml --cpp-compat --output /tmp/sparseir.h .`
  and fails the build on any diff.
- `sparse-ir-capi/assets/sparse_ir_capi.h` is the `cargo-c` distribution copy
  of the same header (`[package.metadata.capi.header] generation = false`
  tells `cargo-c` to use this checked-in file rather than regenerate it).
  The same `header-sync` CI job diffs this file against the generated header
  too, so both copies must be updated together whenever the C API changes.
- `python/pylibsparseir/ctypes_autogen.py` is generated from the header by
  `python/tools/gen_ctypes.py` — regenerate it rather than hand-editing when
  the C API changes.
- `cxx_tests/` exercises the built C API from C++ against the same header;
  `cxx_tests/run_with_rust_capi.sh` builds `sparse-ir-capi` in release mode,
  installs it locally, and builds/runs the CMake test suite against it.

## CI Entry Points

Workflows in `.github/workflows/` (all triggered on push/PR to `main`):

- `rust.yml` — `rust-default` (faer backend) and `rust-system-blas` jobs run
  `cargo build`/`cargo test --all-targets --release --locked` for the
  workspace and for `sparse-ir --features system-blas` respectively; a
  `header-sync` job checks the two C headers described above stay in sync
  with cbindgen output.
- `rust_capi.yml` — builds and tests `sparse-ir-capi` in release mode
  (`cargo build -p sparse-ir-capi --release`, `cargo test -p sparse-ir-capi
  --release`).
- `cxx_tests.yml` — runs the C++ integration tests in `cxx_tests/` against
  the built C API, once with the default (faer) backend and once with
  OpenBLAS (LP64).
- `check_version.yml` — runs `python3 check_version.py`.
- `test_python.yml`, `test_fortran.yml` — test the Python and Fortran
  bindings against the built C API.
- `CI_PublishPyPI.yml`, `PublishPyPI.yml`, `publish_conda.yml`,
  `publish-libsparseir-reusable.yml`, `publish-libsparseir.yml`,
  `manual-release.yml` — publishing workflows; `manual-release.yml` is the
  manual Rust crates.io release flow referenced from `AGENTS.md` and
  `agent-skills/manual-rust-release/SKILL.md`. Do not bypass its
  expected-version and publication-order checks with ad hoc local
  `cargo publish` commands.
- `latest-dependencies.yml` — dependency-drift check.

## Before Editing

- Prefer `cargo build`/`cargo test` scoped to the crate you touched
  (`-p sparse-ir` or `-p sparse-ir-capi`) for fast local iteration; run the
  full workspace and `system-blas` variants before a PR that touches shared
  code (GEMM dispatch, numeric types, or the C API).
- A change to any exported `spir_*` function, struct, enum, or constant in
  `sparse-ir-capi/src/` requires regenerating and re-diffing both C headers
  (see "C API Surface" above) in the same change, or the `header-sync` CI
  job will fail.
- `sparse-ir/*_tests.rs` files are included into their sibling `src/*.rs`
  module; when adding tests for a module that doesn't yet have one, follow
  the existing `mod foo; ... #[cfg(test)] mod foo_tests;`-style pattern
  rather than inlining a large test module.
- Numerical algorithm changes belong in `sparse-ir`; ABI translation,
  pointer/handle validation, and status-code mapping belong in
  `sparse-ir-capi`. See the shared
  [`ffi-boundary.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/ffi-boundary.md)
  and [`numerical-conventions.md`](https://github.com/SpM-lab/spm-agent-rules/blob/main/rules/numerical-conventions.md)
  rules for what each layer must validate and document.
