/**
 * @file sparseir_capi.h
 * @brief C API for SparseIR library
 *
 * This header provides a C-compatible interface to the SparseIR library.
 */

#ifndef SPARSEIR_CAPI_H
#define SPARSEIR_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ============================================================================
 * Error Codes
 * ========================================================================= */

typedef int32_t SparseIRStatus;

#define SPIR_SUCCESS                   0
#define SPIR_ERROR_NULL_POINTER       -1
#define SPIR_ERROR_INVALID_ARGUMENT   -2
#define SPIR_ERROR_PANIC              -99

/* ============================================================================
 * Opaque Types
 * ========================================================================= */

/**
 * @brief Opaque kernel object
 *
 * Represents either a LogisticKernel or RegularizedBoseKernel.
 * Created by spir_kernel_*_new() and destroyed by spir_kernel_release().
 */
typedef struct SparseIRKernel SparseIRKernel;

/* ============================================================================
 * Kernel API
 * ========================================================================= */

/**
 * @brief Create a new Logistic kernel
 *
 * @param lambda Kernel parameter Λ = β * ωmax (must be > 0)
 * @param out Pointer to store the created kernel
 * @return SPIR_SUCCESS on success, error code otherwise
 *
 * @note The caller must release the kernel with spir_kernel_release()
 */
SparseIRStatus spir_kernel_logistic_new(double lambda, SparseIRKernel** out);

/**
 * @brief Create a new RegularizedBose kernel
 *
 * @param lambda Kernel parameter Λ = β * ωmax (must be > 0)
 * @param out Pointer to store the created kernel
 * @return SPIR_SUCCESS on success, error code otherwise
 *
 * @note The caller must release the kernel with spir_kernel_release()
 */
SparseIRStatus spir_kernel_regularized_bose_new(double lambda, SparseIRKernel** out);

/**
 * @brief Release a kernel object
 *
 * @param kernel Kernel to release (can be NULL)
 *
 * @note After calling this function, the kernel pointer is invalid
 */
void spir_kernel_release(SparseIRKernel* kernel);

/**
 * @brief Get the lambda parameter of a kernel
 *
 * @param kernel Kernel object
 * @param out Pointer to store the lambda value
 * @return SPIR_SUCCESS on success, error code otherwise
 */
SparseIRStatus spir_kernel_lambda(const SparseIRKernel* kernel, double* out);

/**
 * @brief Compute kernel value K(x, y)
 *
 * @param kernel Kernel object
 * @param x First argument (typically in [-1, 1])
 * @param y Second argument (typically in [-1, 1])
 * @param out Pointer to store the result
 * @return SPIR_SUCCESS on success, error code otherwise
 */
SparseIRStatus spir_kernel_compute(
    const SparseIRKernel* kernel,
    double x,
    double y,
    double* out
);

#ifdef __cplusplus
}
#endif

#endif /* SPARSEIR_CAPI_H */

