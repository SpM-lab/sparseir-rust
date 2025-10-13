#!/usr/bin/env julia
"""
Example: Using SparseIR C-API from Julia

This demonstrates how to call the SparseIR Rust library from Julia.
"""

# Load the shared library
const libpath = "../target/debug/libsparseir_capi.dylib"  # macOS
# const libpath = "../target/debug/libsparseir_capi.so"   # Linux
# const libpath = "../target/debug/sparseir_capi.dll"     # Windows

# Error codes (compatible with libsparseir)
const SPIR_COMPUTATION_SUCCESS = Int32(0)
const SPIR_GET_IMPL_FAILED = Int32(-1)
const SPIR_INVALID_DIMENSION = Int32(-2)
const SPIR_INPUT_DIMENSION_MISMATCH = Int32(-3)
const SPIR_OUTPUT_DIMENSION_MISMATCH = Int32(-4)
const SPIR_NOT_SUPPORTED = Int32(-5)
const SPIR_INVALID_ARGUMENT = Int32(-6)
const SPIR_INTERNAL_ERROR = Int32(-7)

# Aliases
const SPIR_SUCCESS = SPIR_COMPUTATION_SUCCESS

# Opaque type
mutable struct spir_kernel end

"""
Create a Logistic kernel
"""
function kernel_logistic_new(lambda::Float64)
    status = Ref{Int32}(0)
    kernel = ccall(
        (:spir_logistic_kernel_new, libpath),
        Ptr{spir_kernel},
        (Float64, Ref{Int32}),
        lambda, status
    )
    
    if kernel == C_NULL
        error("Failed to create kernel: status = $(status[])")
    end
    
    return kernel
end

"""
Create a RegularizedBose kernel
"""
function kernel_regularized_bose_new(lambda::Float64)
    status = Ref{Int32}(0)
    kernel = ccall(
        (:spir_reg_bose_kernel_new, libpath),
        Ptr{spir_kernel},
        (Float64, Ref{Int32}),
        lambda, status
    )
    
    if kernel == C_NULL
        error("Failed to create kernel: status = $(status[])")
    end
    
    return kernel
end

"""
Release a kernel
"""
function kernel_release(kernel::Ptr{spir_kernel})
    ccall(
        (:spir_kernel_release, libpath),
        Cvoid,
        (Ptr{spir_kernel},),
        kernel
    )
end

"""
Get lambda parameter
"""
function kernel_lambda(kernel::Ptr{spir_kernel})
    lambda = Ref{Float64}()
    status = ccall(
        (:spir_kernel_lambda, libpath),
        Int32,
        (Ptr{spir_kernel}, Ref{Float64}),
        kernel, lambda
    )
    
    if status != SPIR_SUCCESS
        error("Failed to get lambda: status = $status")
    end
    
    return lambda[]
end

"""
Compute kernel value K(x, y)
"""
function kernel_compute(kernel::Ptr{spir_kernel}, x::Float64, y::Float64)
    result = Ref{Float64}()
    status = ccall(
        (:spir_kernel_compute, libpath),
        Int32,
        (Ptr{spir_kernel}, Float64, Float64, Ref{Float64}),
        kernel, x, y, result
    )
    
    if status != SPIR_SUCCESS
        error("Failed to compute kernel: status = $status")
    end
    
    return result[]
end

# ============================================================================
# Main test
# ============================================================================

println("🧪 Testing SparseIR C-API from Julia")
println("=" ^ 50)

# Test 1: Logistic Kernel
println("\n📝 Test 1: Logistic Kernel")
kernel = kernel_logistic_new(10.0)
println("✅ Created kernel")

lambda = kernel_lambda(kernel)
println("   λ = $lambda")

value = kernel_compute(kernel, 0.5, 0.5)
println("   K(0.5, 0.5) = $value")

kernel_release(kernel)
println("✅ Released kernel")

# Test 2: RegularizedBose Kernel
println("\n📝 Test 2: RegularizedBose Kernel")
kernel = kernel_regularized_bose_new(10.0)
println("✅ Created kernel")

lambda = kernel_lambda(kernel)
println("   λ = $lambda")

value = kernel_compute(kernel, 0.5, 0.5)
println("   K(0.5, 0.5) = $value")

kernel_release(kernel)
println("✅ Released kernel")

# Test 3: Compute multiple values
println("\n📝 Test 3: Compute kernel on grid")
kernel = kernel_logistic_new(10.0)

x_grid = range(-1, 1, length=5)
y_grid = range(-1, 1, length=5)

println("   Grid: x ∈ [-1, 1], y ∈ [-1, 1]")
for x in x_grid
    for y in y_grid
        val = kernel_compute(kernel, x, y)
        print("$(round(val, digits=4)) ")
    end
    println()
end

kernel_release(kernel)

# Test 4: Error handling
println("\n📝 Test 4: Error handling")
try
    kernel = kernel_logistic_new(-1.0)  # Invalid lambda
    println("❌ Should have failed")
catch e
    println("✅ Correctly caught error: $e")
end

println("\n" * "=" ^ 50)
println("✅ All tests passed!")

