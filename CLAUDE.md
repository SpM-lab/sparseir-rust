# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is intended for a Rust wrapper around the libsparseir C++ library. Currently, it contains only git submodules for the required C++ dependencies:

- **libsparseir**: C++ library for sparse intermediate representation of correlation functions
- **libxprec**: Fast emulated quadruple (double-double) precision arithmetic library
- **eigen**: Linear algebra template library

The Rust wrapper implementation has not yet been created.

## Current Structure

The repository currently consists of:
- Git submodules for C++ dependencies in `libsparseir/`, `libxprec/`, and `eigen/` directories
- MIT license file
- Initial commit only

## Dependencies and Build System

### C++ Dependencies (submodules)

**libsparseir**:
- CMake-based build system (requires CMake >= 3.10, C++11 compiler)
- Provides basis functions for arbitrary cutoff Λ with full precision
- Optional BLAS support for performance (`-DSPARSEIR_USE_BLAS=ON`)
- Optional ILP64 BLAS support for large matrices (`-DSPARSEIR_USE_ILP64=ON`)
- Build scripts: `build_capi.sh`, `build_fortran.sh`, `build_with_tests.sh`
- Test command: `./build/test/libsparseirtests`

**libxprec**:
- Header-only or compiled library for double-double precision arithmetic
- CMake build with optional testing (`-DBUILD_TESTING=ON` requires GNU MPFR)
- Recommended flags: `-DCMAKE_CXX_FLAGS=-mfma` for FMA instruction optimization

**eigen**:
- Header-only template library for linear algebra
- No build required, used as dependency by libsparseir

### Future Rust Development Commands

When Rust implementation is added, typical commands will likely include:
```bash
# Build Rust wrapper
cargo build

# Run tests
cargo test

# Build with release optimizations
cargo build --release

# Check code
cargo check

# Format code
cargo fmt

# Run clippy lints
cargo clippy
```

## Architecture Notes

**libsparseir Core Functionality**:
- Intermediate representation (IR) basis construction for fermionic/bosonic correlators
- On-the-fly computation of basis functions with configurable precision
- Sparse sampling algorithms for efficient representation
- BLAS-optimized matrix operations for performance-critical applications

**Integration Strategy**:
When implementing the Rust wrapper, consider:
- FFI bindings to the C API provided by libsparseir
- Safe Rust abstractions over the C interface
- Integration with existing Rust scientific computing ecosystem (ndarray, etc.)
- Proper handling of libxprec's double-double precision types

## Development Environment

- Requires C++11-compliant compiler for building C++ dependencies
- CMake >= 3.10 for building submodules
- Optional: Fortran compiler for Fortran bindings in libsparseir
- Optional: GNU MPFR for extended testing of libxprec