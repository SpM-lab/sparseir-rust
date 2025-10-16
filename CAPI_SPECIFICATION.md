# libsparseir C-API Complete Specification

**Version**: 0.6.0  
**Compatibility**: 100% backward compatibility required  
**Status**: Reference implementation in C++ (libsparseir)  
**Target**: Rust reimplementation in sparseir-capi  

---

## Table of Contents

1. [Overview](#overview)
2. [Opaque Types](#opaque-types)
3. [Constants](#constants)
4. [API Categories](#api-categories)
5. [Function Reference](#function-reference)
6. [Memory Management](#memory-management)
7. [Error Handling](#error-handling)
8. [Implementation Checklist](#implementation-checklist)

---

## Overview

The libsparseir C-API provides a complete interface for:
- Sparse intermediate representation (IR) of imaginary-time Green's functions
- Discrete Lehmann representation (DLR)
- Singular value expansion (SVE) of analytical continuation kernels
- Tau and Matsubara frequency sampling

**Total API Surface**:
- **5 opaque types**
- **8 status codes**
- **11 constants** (statistics, order, working precision, SVD strategy)
- **56 C functions** (excluding internal GEMM registration - not in public API)

---

## Opaque Types

All internal data structures are hidden behind opaque pointers. Each type has:
- `spir_<type>_release()`: Destroy/free the object
- `spir_<type>_clone()`: Create a copy
- `spir_<type>_is_assigned()`: Check if valid
- `_spir_<type>_get_raw_ptr()`: Debug access (internal)

### Type List

```c
struct _spir_kernel;
typedef struct _spir_kernel spir_kernel;

struct _spir_funcs;
typedef struct _spir_funcs spir_funcs;

struct _spir_basis;
typedef struct _spir_basis spir_basis;

struct _spir_sampling;
typedef struct _spir_sampling spir_sampling;

struct _spir_sve_result;
typedef struct _spir_sve_result spir_sve_result;
```

### Memory Management Pattern

```c
// Create
spir_kernel* k = spir_logistic_kernel_new(lambda, &status);

// Clone (increment reference count or deep copy)
spir_kernel* k2 = spir_kernel_clone(k);

// Check validity
int is_valid = spir_kernel_is_assigned(k);

// Release (decrement reference count, free if zero)
spir_kernel_release(k);
spir_kernel_release(k2);
```

---

## Constants

### Status Codes

| Name | Value | Description |
|------|-------|-------------|
| `SPIR_COMPUTATION_SUCCESS` | 0 | Success |
| `SPIR_GET_IMPL_FAILED` | -1 | Failed to get implementation |
| `SPIR_INVALID_DIMENSION` | -2 | Invalid dimension |
| `SPIR_INPUT_DIMENSION_MISMATCH` | -3 | Input dimension mismatch |
| `SPIR_OUTPUT_DIMENSION_MISMATCH` | -4 | Output dimension mismatch |
| `SPIR_NOT_SUPPORTED` | -5 | Operation not supported |
| `SPIR_INVALID_ARGUMENT` | -6 | Invalid argument |
| `SPIR_INTERNAL_ERROR` | -7 | Internal error |

**Alias**: `SPIR_SUCCESS` = `SPIR_COMPUTATION_SUCCESS`

### Statistics Type

| Name | Value | Description |
|------|-------|-------------|
| `SPIR_STATISTICS_FERMIONIC` | 1 | Fermionic (anti-periodic) |
| `SPIR_STATISTICS_BOSONIC` | 0 | Bosonic (periodic) |

### Memory Order

| Name | Value | Description |
|------|-------|-------------|
| `SPIR_ORDER_ROW_MAJOR` | 0 | C-order (last index varies fastest) |
| `SPIR_ORDER_COLUMN_MAJOR` | 1 | Fortran-order (first index varies fastest) |

### Working Precision (Twork)

| Name | Value | Description |
|------|-------|-------------|
| `SPIR_TWORK_FLOAT64` | 0 | Double precision (f64) |
| `SPIR_TWORK_FLOAT64X2` | 1 | Extended precision (f64x2 / TwoFloat) |
| `SPIR_TWORK_AUTO` | -1 | Automatic selection based on epsilon |

### SVD Strategy

| Name | Value | Description |
|------|-------|-------------|
| `SPIR_SVDSTRAT_FAST` | 0 | Fast SVD (less accurate) |
| `SPIR_SVDSTRAT_ACCURATE` | 1 | Accurate SVD (slower) |
| `SPIR_SVDSTRAT_AUTO` | -1 | Automatic selection |

---

## API Categories

### 1. Kernel Operations (3 functions)

Kernel functions define the analytical continuation kernels.

| Function | Purpose |
|----------|---------|
| `spir_logistic_kernel_new` | Create logistic kernel (fermionic/bosonic) |
| `spir_reg_bose_kernel_new` | Create regularized bosonic kernel |
| `spir_kernel_domain` | Get kernel domain boundaries |

**Plus 4 standard opaque functions**: `_release`, `_clone`, `_is_assigned`, `_get_raw_ptr`

### 2. SVE Operations (3 functions)

Singular Value Expansion of kernels.

| Function | Purpose |
|----------|---------|
| `spir_sve_result_new` | Compute SVE of a kernel |
| `spir_sve_result_truncate` | Truncate SVE result |
| `spir_sve_result_get_size` | Get number of singular values |
| `spir_sve_result_get_svals` | Get singular values array |

**Plus 4 standard opaque functions**

### 3. Funcs Operations (9 functions)

Function objects (basis functions in different domains).

| Function | Purpose |
|----------|---------|
| `spir_funcs_get_size` | Get number of functions |
| `spir_funcs_get_slice` | Create subset of functions |
| `spir_funcs_eval` | Evaluate at single point (continuous) |
| `spir_funcs_eval_matsu` | Evaluate at single Matsubara frequency |
| `spir_funcs_batch_eval` | Evaluate at multiple points (continuous) |
| `spir_funcs_batch_eval_matsu` | Evaluate at multiple Matsubara frequencies |
| `spir_funcs_get_n_knots` | Get number of knots (piecewise poly) |
| `spir_funcs_get_knots` | Get knot positions (piecewise poly) |

**Plus 4 standard opaque functions**

### 4. Basis Operations (20 functions)

IR and DLR basis construction and accessors.

#### Basis Construction (1 function)

| Function | Purpose |
|----------|---------|
| `spir_basis_new` | Create IR basis from SVE result |

#### Basis Properties (5 functions)

| Function | Purpose |
|----------|---------|
| `spir_basis_get_size` | Get number of basis functions |
| `spir_basis_get_svals` | Get singular values (deprecated) |
| `spir_basis_get_singular_values` | Get singular values (preferred) |
| `spir_basis_get_stats` | Get statistics type |

#### Basis Function Accessors (3 functions)

| Function | Purpose |
|----------|---------|
| `spir_basis_get_u` | Get u basis functions (imaginary time) |
| `spir_basis_get_v` | Get v basis functions (real frequency) |
| `spir_basis_get_uhat` | Get û basis functions (Matsubara) |

#### Default Sampling Points (7 functions)

| Function | Purpose |
|----------|---------|
| `spir_basis_get_n_default_taus` | Get number of default τ points |
| `spir_basis_get_default_taus` | Get default τ sampling points |
| `spir_basis_get_default_taus_ext` | Get extended τ sampling points |
| `spir_basis_get_n_default_ws` | Get number of default ω points |
| `spir_basis_get_default_ws` | Get default ω sampling points |
| `spir_basis_get_n_default_matsus` | Get number of default Matsubara points |
| `spir_basis_get_default_matsus` | Get default Matsubara points |
| `spir_basis_get_n_default_matsus_ext` | Get extended Matsubara points count |
| `spir_basis_get_default_matsus_ext` | Get extended Matsubara points |

**Plus 4 standard opaque functions**

### 5. DLR Operations (7 functions)

Discrete Lehmann Representation.

| Function | Purpose |
|----------|---------|
| `spir_dlr_new` | Create DLR from IR basis (automatic poles) |
| `spir_dlr_new_with_poles` | Create DLR with custom poles |
| `spir_dlr_get_npoles` | Get number of poles |
| `spir_dlr_get_poles` | Get pole positions |
| `spir_ir2dlr_dd` | Transform IR→DLR (real) |
| `spir_ir2dlr_zz` | Transform IR→DLR (complex) |
| `spir_dlr2ir_dd` | Transform DLR→IR (real) |
| `spir_dlr2ir_zz` | Transform DLR→IR (complex) |

**Note**: DLR objects use the `spir_basis` type, but represent DLR instead of IR.

### 6. Sampling Operations (14 functions)

Sparse sampling in τ and Matsubara domains.

#### Sampling Construction (4 functions)

| Function | Purpose |
|----------|---------|
| `spir_tau_sampling_new` | Create τ sampling with custom points |
| `spir_tau_sampling_new_with_matrix` | Create τ sampling with matrix |
| `spir_matsu_sampling_new` | Create Matsubara sampling |
| `spir_matsu_sampling_new_with_matrix` | Create Matsubara sampling with matrix |

#### Sampling Properties (4 functions)

| Function | Purpose |
|----------|---------|
| `spir_sampling_get_npoints` | Get number of sampling points |
| `spir_sampling_get_taus` | Get τ sampling points |
| `spir_sampling_get_matsus` | Get Matsubara frequency indices |
| `spir_sampling_get_cond_num` | Get condition number of matrix |

#### Sampling Operations (6 functions)

| Function | Purpose |
|----------|---------|
| `spir_sampling_eval_dd` | Evaluate (double → double) |
| `spir_sampling_eval_dz` | Evaluate (double → complex) |
| `spir_sampling_eval_zz` | Evaluate (complex → complex) |
| `spir_sampling_fit_dd` | Fit (double → double) |
| `spir_sampling_fit_zd` | Fit (complex → double) |
| `spir_sampling_fit_zz` | Fit (complex → complex) |

**Plus 4 standard opaque functions**

---

## Function Reference

### Detailed Signatures

#### Kernel

```c
// Create logistic kernel
spir_kernel* spir_logistic_kernel_new(double lambda, int* status);

// Create regularized bosonic kernel
spir_kernel* spir_reg_bose_kernel_new(double lambda, int* status);

// Get kernel domain
int spir_kernel_domain(const spir_kernel* k, 
                       double* xmin, double* xmax,
                       double* ymin, double* ymax);
```

#### SVE

```c
// Compute SVE
spir_sve_result* spir_sve_result_new(
    const spir_kernel* k,
    double epsilon,
    double cutoff,        // -1 for default
    int lmax,
    int n_gauss,
    int Twork,            // SPIR_TWORK_*
    int* status
);

// Truncate SVE
spir_sve_result* spir_sve_result_truncate(
    const spir_sve_result* sve,
    double epsilon,
    int max_size,         // -1 for no limit
    int* status
);

// Get SVE size
int spir_sve_result_get_size(const spir_sve_result* sve, int* size);

// Get singular values
int spir_sve_result_get_svals(const spir_sve_result* sve, double* svals);
```

#### Funcs

```c
// Get number of functions
int spir_funcs_get_size(const spir_funcs* funcs, int* size);

// Get slice
spir_funcs* spir_funcs_get_slice(const spir_funcs* funcs, 
                                 int nslice, 
                                 int* indices, 
                                 int* status);

// Evaluate at single point (continuous domain)
int spir_funcs_eval(const spir_funcs* funcs, double x, double* out);

// Evaluate at single Matsubara frequency
int spir_funcs_eval_matsu(const spir_funcs* funcs, int64_t n, c_complex* out);

// Batch evaluate (continuous domain)
int spir_funcs_batch_eval(const spir_funcs* funcs, 
                          int order,          // SPIR_ORDER_*
                          int num_points, 
                          const double* xs, 
                          double* out);

// Batch evaluate (Matsubara)
int spir_funcs_batch_eval_matsu(const spir_funcs* funcs, 
                                int order, 
                                int num_freqs,
                                const int64_t* matsubara_freq_indices, 
                                c_complex* out);

// Get number of knots
int spir_funcs_get_n_knots(const spir_funcs* funcs, int* n_knots);

// Get knots
int spir_funcs_get_knots(const spir_funcs* funcs, double* knots);
```

#### Basis

```c
// Create IR basis
spir_basis* spir_basis_new(
    int statistics,       // SPIR_STATISTICS_*
    double beta,
    double omega_max,
    double epsilon,
    const spir_kernel* k,
    const spir_sve_result* sve,
    int max_size,         // -1 for no limit
    int* status
);

// Get basis size
int spir_basis_get_size(const spir_basis* b, int* size);

// Get singular values (deprecated)
int spir_basis_get_svals(const spir_basis* b, double* svals);

// Get singular values (preferred)
int spir_basis_get_singular_values(const spir_basis* b, double* svals);

// Get statistics type
int spir_basis_get_stats(const spir_basis* b, int* statistics);

// Get u basis functions (imaginary time)
spir_funcs* spir_basis_get_u(const spir_basis* b, int* status);

// Get v basis functions (real frequency)
spir_funcs* spir_basis_get_v(const spir_basis* b, int* status);

// Get û basis functions (Matsubara)
spir_funcs* spir_basis_get_uhat(const spir_basis* b, int* status);

// Default τ sampling
int spir_basis_get_n_default_taus(const spir_basis* b, int* num_points);
int spir_basis_get_default_taus(const spir_basis* b, double* points);
int spir_basis_get_default_taus_ext(const spir_basis* b, 
                                    int n_points, 
                                    double* points,
                                    int* n_points_returned);

// Default ω sampling
int spir_basis_get_n_default_ws(const spir_basis* b, int* num_points);
int spir_basis_get_default_ws(const spir_basis* b, double* points);

// Default Matsubara sampling
int spir_basis_get_n_default_matsus(const spir_basis* b, 
                                    bool positive_only,
                                    int* num_points);
int spir_basis_get_default_matsus(const spir_basis* b, 
                                  bool positive_only,
                                  int64_t* points);
int spir_basis_get_n_default_matsus_ext(const spir_basis* b,
                                        bool positive_only,
                                        int L,
                                        int* num_points_returned);
int spir_basis_get_default_matsus_ext(const spir_basis* b,
                                      bool positive_only,
                                      int n_points,
                                      int64_t* points,
                                      int* n_points_returned);
```

#### DLR

```c
// Create DLR with automatic poles
spir_basis* spir_dlr_new(const spir_basis* b, int* status);

// Create DLR with custom poles
spir_basis* spir_dlr_new_with_poles(const spir_basis* b,
                                    const int npoles,
                                    const double* poles,
                                    int* status);

// Get number of poles
int spir_dlr_get_npoles(const spir_basis* dlr, int* num_poles);

// Get poles
int spir_dlr_get_poles(const spir_basis* dlr, double* poles);

// IR → DLR transformations
int spir_ir2dlr_dd(const spir_basis* dlr,
                   int order,
                   int ndim,
                   const int* input_dims,
                   int target_dim,
                   const double* input,
                   double* out);

int spir_ir2dlr_zz(const spir_basis* dlr,
                   int order,
                   int ndim,
                   const int* input_dims,
                   int target_dim,
                   const c_complex* input,
                   c_complex* out);

// DLR → IR transformations
int spir_dlr2ir_dd(const spir_basis* dlr,
                   int order,
                   int ndim,
                   const int* input_dims,
                   int target_dim,
                   const double* input,
                   double* out);

int spir_dlr2ir_zz(const spir_basis* dlr,
                   int order,
                   int ndim,
                   const int* input_dims,
                   int target_dim,
                   const c_complex* input,
                   c_complex* out);
```

#### Sampling

```c
// Create tau sampling
spir_sampling* spir_tau_sampling_new(const spir_basis* b,
                                     int num_points,
                                     const double* points,
                                     int* status);

spir_sampling* spir_tau_sampling_new_with_matrix(int order,
                                                 int statistics,
                                                 int basis_size,
                                                 int num_points,
                                                 const double* points,
                                                 const double* matrix,
                                                 int* status);

// Create Matsubara sampling
spir_sampling* spir_matsu_sampling_new(const spir_basis* b,
                                       bool positive_only,
                                       int num_points,
                                       const int64_t* points,
                                       int* status);

spir_sampling* spir_matsu_sampling_new_with_matrix(int order,
                                                   int statistics,
                                                   int basis_size,
                                                   bool positive_only,
                                                   int num_points,
                                                   const int64_t* points,
                                                   const c_complex* matrix,
                                                   int* status);

// Get sampling properties
int spir_sampling_get_npoints(const spir_sampling* s, int* num_points);
int spir_sampling_get_taus(const spir_sampling* s, double* points);
int spir_sampling_get_matsus(const spir_sampling* s, int64_t* points);
int spir_sampling_get_cond_num(const spir_sampling* s, double* cond_num);

// Evaluate at sampling points
int spir_sampling_eval_dd(const spir_sampling* s,
                          int order,
                          int ndim,
                          const int* input_dims,
                          int target_dim,
                          const double* input,
                          double* out);

int spir_sampling_eval_dz(const spir_sampling* s,
                          int order,
                          int ndim,
                          const int* input_dims,
                          int target_dim,
                          const double* input,
                          c_complex* out);

int spir_sampling_eval_zz(const spir_sampling* s,
                          int order,
                          int ndim,
                          const int* input_dims,
                          int target_dim,
                          const c_complex* input,
                          c_complex* out);

// Fit from sampling points
int spir_sampling_fit_dd(const spir_sampling* s,
                         int order,
                         int ndim,
                         const int* input_dims,
                         int target_dim,
                         const double* input,
                         double* out);

int spir_sampling_fit_zd(const spir_sampling* s,
                         int order,
                         int ndim,
                         const int* input_dims,
                         int target_dim,
                         const c_complex* input,
                         double* out);

int spir_sampling_fit_zz(const spir_sampling* s,
                         int order,
                         int ndim,
                         const int* input_dims,
                         int target_dim,
                         const c_complex* input,
                         c_complex* out);
```

---

## Memory Management

### Reference Counting Pattern

All opaque types use **shared ownership** (similar to `std::shared_ptr` in C++ or `Arc` in Rust):

1. **Constructor**: Returns a new object with reference count = 1
2. **Clone**: Increments reference count, returns pointer to same object
3. **Release**: Decrements reference count, frees when count reaches 0

```c
// Example
spir_kernel* k1 = spir_logistic_kernel_new(1000.0, &status);  // refcount = 1
spir_kernel* k2 = spir_kernel_clone(k1);                      // refcount = 2

spir_kernel_release(k1);  // refcount = 1, k1 pointer now invalid
spir_kernel_release(k2);  // refcount = 0, memory freed
```

### Array Output Pattern

Functions that return arrays use **pre-allocated buffers**:

```c
// 1. Get size
int size;
spir_basis_get_size(basis, &size);

// 2. Allocate buffer
double* svals = (double*)malloc(size * sizeof(double));

// 3. Fill buffer
spir_basis_get_singular_values(basis, svals);

// 4. Use data
for (int i = 0; i < size; i++) {
    printf("sval[%d] = %f\n", i, svals[i]);
}

// 5. Free buffer
free(svals);
```

### Multi-dimensional Arrays

For ND arrays, use **flat layout** with explicit order:

```c
// Example: 2D array (3x5)
int ndim = 2;
int dims[2] = {3, 5};
int target_dim = 1;  // Transform along dimension 1

// Row-major (C order): shape (3, 5) → flat index [i*5 + j]
double input_row[15];
double output_row[15];
spir_sampling_eval_dd(s, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim, input_row, output_row);

// Column-major (Fortran order): shape (3, 5) → flat index [i + j*3]
double input_col[15];
double output_col[15];
spir_sampling_eval_dd(s, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim, input_col, output_col);
```

---

## Error Handling

### Status Code Pattern

Most functions return:
- **0 (SPIR_COMPUTATION_SUCCESS)** on success
- **Negative value** on error

For functions that return pointers:
- **Non-NULL** on success
- **NULL** on error, with status set via output parameter

```c
int status;

// Pattern 1: Function returns pointer
spir_kernel* k = spir_logistic_kernel_new(1000.0, &status);
if (status != SPIR_COMPUTATION_SUCCESS) {
    fprintf(stderr, "Error: %d\n", status);
    return -1;
}

// Pattern 2: Function returns status
int result = spir_basis_get_size(basis, &size);
if (result != SPIR_COMPUTATION_SUCCESS) {
    fprintf(stderr, "Error: %d\n", result);
    return -1;
}
```

### Common Error Scenarios

| Error Code | Common Causes |
|------------|---------------|
| `SPIR_GET_IMPL_FAILED` | Object type mismatch (e.g., calling DLR function on IR basis) |
| `SPIR_INVALID_DIMENSION` | Array dimension out of range |
| `SPIR_INPUT_DIMENSION_MISMATCH` | Input array size doesn't match expected size |
| `SPIR_OUTPUT_DIMENSION_MISMATCH` | Output array size doesn't match expected size |
| `SPIR_NOT_SUPPORTED` | Operation not supported for this object type |
| `SPIR_INVALID_ARGUMENT` | Invalid parameter value (e.g., negative beta) |
| `SPIR_INTERNAL_ERROR` | Unexpected internal error |

---

## Implementation Checklist

### sparseir-capi Implementation Status

#### Kernel (7/7 complete ✅)
- [x] `spir_logistic_kernel_new`
- [x] `spir_reg_bose_kernel_new`
- [x] `spir_kernel_domain`
- [x] `spir_kernel_release` (via `impl_opaque_type_common!` macro)
- [x] `spir_kernel_clone` (via macro)
- [x] `spir_kernel_is_assigned` (via macro)
- [x] `_spir_kernel_get_raw_ptr` (via macro)

#### SVE (7/8 complete)
- [x] `spir_sve_result_new`
- [ ] `spir_sve_result_truncate` ⚠️ **ONLY MISSING FUNCTION**
- [x] `spir_sve_result_get_size`
- [x] `spir_sve_result_get_svals`
- [x] `spir_sve_result_release` (via macro)
- [x] `spir_sve_result_clone` (via macro)
- [x] `spir_sve_result_is_assigned` (via macro)
- [x] `_spir_sve_result_get_raw_ptr` (via macro)

#### Funcs (13/13 complete ✅)
- [x] `spir_funcs_get_size`
- [x] `spir_funcs_get_slice`
- [x] `spir_funcs_eval`
- [x] `spir_funcs_eval_matsu`
- [x] `spir_funcs_batch_eval`
- [x] `spir_funcs_batch_eval_matsu`
- [x] `spir_funcs_get_n_knots`
- [x] `spir_funcs_get_knots`
- [x] `spir_funcs_release` (via macro)
- [x] `spir_funcs_clone` (via macro)
- [x] `spir_funcs_is_assigned` (via macro)
- [x] `_spir_funcs_get_raw_ptr` (via macro)

#### Basis (24/24 complete ✅)
- [x] `spir_basis_new`
- [x] `spir_basis_get_size`
- [x] `spir_basis_get_svals`
- [x] `spir_basis_get_singular_values`
- [x] `spir_basis_get_stats`
- [x] `spir_basis_get_u`
- [x] `spir_basis_get_v`
- [x] `spir_basis_get_uhat`
- [x] `spir_basis_get_n_default_taus`
- [x] `spir_basis_get_default_taus`
- [x] `spir_basis_get_default_taus_ext`
- [x] `spir_basis_get_n_default_ws`
- [x] `spir_basis_get_default_ws`
- [x] `spir_basis_get_n_default_matsus`
- [x] `spir_basis_get_default_matsus`
- [x] `spir_basis_get_n_default_matsus_ext`
- [x] `spir_basis_get_default_matsus_ext`
- [x] `spir_basis_release` (via macro)
- [x] `spir_basis_clone` (via macro)
- [x] `spir_basis_is_assigned` (via macro)
- [x] `_spir_basis_get_raw_ptr` (via macro)

#### DLR (7/7 complete)
- [x] `spir_dlr_new`
- [x] `spir_dlr_new_with_poles`
- [x] `spir_dlr_get_npoles`
- [x] `spir_dlr_get_poles`
- [x] `spir_ir2dlr_dd`
- [x] `spir_ir2dlr_zz`
- [x] `spir_dlr2ir_dd`
- [x] `spir_dlr2ir_zz`

#### Sampling (18/18 complete ✅)
- [x] `spir_tau_sampling_new`
- [x] `spir_tau_sampling_new_with_matrix`
- [x] `spir_matsu_sampling_new`
- [x] `spir_matsu_sampling_new_with_matrix`
- [x] `spir_sampling_get_npoints`
- [x] `spir_sampling_get_taus`
- [x] `spir_sampling_get_matsus`
- [x] `spir_sampling_get_cond_num`
- [x] `spir_sampling_eval_dd`
- [x] `spir_sampling_eval_dz`
- [x] `spir_sampling_eval_zz`
- [x] `spir_sampling_fit_dd`
- [x] `spir_sampling_fit_zd`
- [x] `spir_sampling_fit_zz`
- [x] `spir_sampling_release` (via macro)
- [x] `spir_sampling_clone` (via macro)
- [x] `spir_sampling_is_assigned` (via macro)
- [x] `_spir_sampling_get_raw_ptr` (via macro)

### Summary

**Total**: 56 core functions + 20 opaque type management = **76 functions**

**Implemented**: 
- Core: 55/56 (98%) - **Only `spir_sve_result_truncate` missing**
- Opaque management: 20/20 (100%) - **All via `impl_opaque_type_common!` macro**
- **Overall**: 75/76 (99%)

### 🎯 For 100% Compatibility

**Only 1 function remaining**: `spir_sve_result_truncate`

**Priority for 100% compatibility**:
1. ✅ ~~Implement all 20 opaque type management functions~~ (DONE via macro)
2. ⚠️ Implement `spir_sve_result_truncate` (ONLY REMAINING)
3. ⏳ Comprehensive C-API tests

---

## Notes

### Complex Number Handling

C99 complex vs C++ complex:
```c
#if defined(_MSC_VER) || defined(__cplusplus)
    #include <complex>
    typedef std::complex<double> c_complex;
#else
    #include <complex.h>
    typedef double _Complex c_complex;
#endif
```

Rust FFI must handle both representations.

### Matsubara Frequency Convention

- **Fermionic**: ωn = (2n+1)π/β (odd indices)
- **Bosonic**: ωn = 2nπ/β (even indices)

Integer `n` is passed, not the actual frequency.

### Memory Order Implications

- **Row-major** (C): `array[i][j]` → flat index `i * ncols + j`
- **Column-major** (Fortran): `array[i, j]` → flat index `i + j * nrows`

The `order` parameter affects interpretation of the flat buffer.

---

## Version History

- **0.6.0** (Current): Full API as documented
- **0.5.x**: Earlier versions (not covered)

---

## References

- **Header**: `libsparseir/include/sparseir/sparseir.h`
- **Status codes**: `libsparseir/include/sparseir/spir_status.h`
- **Version**: `libsparseir/include/sparseir/version.h`
- **Implementation**: `sparseir-capi/src/*.rs`

