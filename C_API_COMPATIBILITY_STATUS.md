# C-API Compatibility Status with libsparseir

This document tracks the implementation status of all functions in `libsparseir/include/sparseir/sparseir.h`.

**Last Updated**: 2025-10-14

## Summary

| Module | Total | Implemented | Status |
|--------|-------|-------------|--------|
| Kernel API | 5 | 5 | ✅ 100% |
| SVE API | 5 | 5 | ✅ 100% |
| Basis API | 18 | 15 | 🔄 83% |
| Funcs API | 10 | 1 | 🔄 10% |
| DLR API | 8 | 0 | ❌ 0% |
| Sampling API | 15 | 0 | ❌ 0% |
| **TOTAL** | **61** | **26** | **43%** |

*Note: register_gemm functions excluded as per requirements*

---

## Kernel API ✅ (5/5 - 100%)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_logistic_kernel_new` | ✅ | |
| `spir_reg_bose_kernel_new` | ✅ | |
| `spir_kernel_lambda` | ✅ | |
| `spir_kernel_compute` | ✅ | |
| `spir_kernel_release` | ✅ | Auto-generated |

---

## SVE API ✅ (5/5 - 100%)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_sve_result_new` | ✅ | |
| `spir_sve_result_get_size` | ✅ | |
| `spir_sve_result_get_svals` | ✅ | |
| `spir_sve_result_truncate` | ✅ | |
| `spir_sve_result_release` | ✅ | Auto-generated |

---

## Basis API 🔄 (15/18 - 83%)

### Core Functions ✅ (15/15)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_basis_new` | ✅ | |
| `spir_basis_release` | ✅ | |
| `spir_basis_get_size` | ✅ | |
| `spir_basis_get_svals` | ✅ | |
| `spir_basis_get_singular_values` | ✅ | Alias for get_svals |
| `spir_basis_get_stats` | ✅ | |
| `spir_basis_get_n_default_taus` | ✅ | |
| `spir_basis_get_default_taus` | ✅ | |
| `spir_basis_get_n_default_ws` | ✅ | |
| `spir_basis_get_default_ws` | ✅ | |
| `spir_basis_get_n_default_matsus` | ✅ | |
| `spir_basis_get_default_matsus` | ✅ | |
| `spir_basis_get_u` | ✅ | |
| `spir_basis_get_v` | ✅ | |
| `spir_basis_get_uhat` | ✅ | |

### Extended Functions ❌ (0/3)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_basis_get_default_taus_ext` | ❌ | Extended tau sampling |
| `spir_basis_get_n_default_matsus_ext` | ❌ | Extended matsubara count |
| `spir_basis_get_default_matsus_ext` | ❌ | Extended matsubara sampling |

---

## Funcs API 🔄 (1/10 - 10%)

### Basic Management ✅ (1/1)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_funcs_release` | ✅ | |

### Introspection ❌ (0/3)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_funcs_get_size` | ❌ | Get number of basis functions |
| `spir_funcs_get_n_knots` | ❌ | Get number of knots |
| `spir_funcs_get_knots` | ❌ | Get knot positions |

### Operations ❌ (0/2)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_funcs_get_slice` | ❌ | Extract subset of functions |
| *(clone, is_assigned)* | ❌ | Auto-generated helper functions |

### Evaluation ❌ (0/4)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_funcs_eval` | ❌ | Eval at single point (tau) |
| `spir_funcs_batch_eval` | ❌ | Eval at multiple points (tau) |
| `spir_funcs_eval_matsu` | ❌ | Eval at single Matsubara freq |
| `spir_funcs_batch_eval_matsu` | ❌ | Eval at multiple Matsubara freqs |

---

## DLR API ❌ (0/8 - 0%)

### Construction ❌ (0/2)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_dlr_new` | ❌ | Create DLR from basis |
| `spir_dlr_new_with_poles` | ❌ | Create DLR with custom poles |

### Introspection ❌ (0/2)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_dlr_get_npoles` | ❌ | Get number of poles |
| `spir_dlr_get_poles` | ❌ | Get pole positions |

### Transformation ❌ (0/4)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_ir2dlr_dd` | ❌ | IR→DLR (real coeffs) |
| `spir_ir2dlr_zz` | ❌ | IR→DLR (complex coeffs) |
| `spir_dlr2ir_dd` | ❌ | DLR→IR (real coeffs) |
| `spir_dlr2ir_zz` | ❌ | DLR→IR (complex coeffs) |

---

## Sampling API ❌ (0/15 - 0%)

### Construction ❌ (0/4)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_tau_sampling_new` | ❌ | Create tau sampling |
| `spir_tau_sampling_new_with_matrix` | ❌ | Create tau sampling with custom matrix |
| `spir_matsu_sampling_new` | ❌ | Create Matsubara sampling |
| `spir_matsu_sampling_new_with_matrix` | ❌ | Create Matsubara with custom matrix |

### Introspection ❌ (0/4)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_sampling_get_npoints` | ❌ | Get number of sampling points |
| `spir_sampling_get_taus` | ❌ | Get tau sampling points |
| `spir_sampling_get_matsus` | ❌ | Get Matsubara sampling points |
| `spir_sampling_get_cond_num` | ❌ | Get condition number |

### Evaluation ❌ (0/3)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_sampling_eval_dd` | ❌ | Eval (real data → real) |
| `spir_sampling_eval_dz` | ❌ | Eval (real data → complex) |
| `spir_sampling_eval_zz` | ❌ | Eval (complex data → complex) |

### Fitting ❌ (0/3)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_sampling_fit_dd` | ❌ | Fit (real data → real) |
| `spir_sampling_fit_zd` | ❌ | Fit (complex data → real) |
| `spir_sampling_fit_zz` | ❌ | Fit (complex data → complex) |

### Management ❌ (0/1)

| Function | Status | Notes |
|----------|--------|-------|
| `spir_sampling_release` | ❌ | |

---

## Implementation Priority

1. **Funcs API (Phase 2)** - Enable function evaluation (9 functions)
2. **Basis API Extensions** - Complete basis API (3 functions)
3. **Sampling API** - High-level operations (15 functions)
4. **DLR API** - DLR transformation (8 functions)

**Next Target**: Funcs API evaluation functions to make basis functions usable.

