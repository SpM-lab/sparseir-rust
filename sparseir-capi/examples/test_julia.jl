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

# Test 5: SVE computation
println("\n📝 Test 5: SVE Computation")
kernel = kernel_logistic_new(10.0)

# Compute SVE
status = Ref{Int32}(0)
sve = ccall(
    (:spir_sve_result_new, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Float64, Float64, Int32, Int32, Int32, Ref{Int32}),
    kernel, 1e-6, -1.0, -1, -1, -1, status
)

if sve == C_NULL || status[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to compute SVE: status = $(status[])")
end

# Get SVE size
size = Ref{Int32}(0)
status_size = ccall(
    (:spir_sve_result_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    sve, size
)

if status_size != SPIR_COMPUTATION_SUCCESS
    error("Failed to get SVE size: status = $status_size")
end

println("✅ Computed SVE")
println("   Size: $(size[])")

# Get singular values
svals = Vector{Float64}(undef, size[])
status_svals = ccall(
    (:spir_sve_result_get_svals, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    sve, svals
)

if status_svals != SPIR_COMPUTATION_SUCCESS
    error("Failed to get singular values: status = $status_svals")
end

println("   First 5 singular values:")
for i in 1:min(5, length(svals))
    println("     s[$i] = $(svals[i])")
end

# Test truncation
status_truncate = Ref{Int32}(0)
sve_truncated = ccall(
    (:spir_sve_result_truncate, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Float64, Int32, Ref{Int32}),
    sve, 1e-4, div(size[], 2), status_truncate
)

if sve_truncated == C_NULL || status_truncate[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to truncate SVE: status = $(status_truncate[])")
end

size_truncated = Ref{Int32}(0)
ccall(
    (:spir_sve_result_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    sve_truncated, size_truncated
)

println("   Truncated size: $(size_truncated[])")

# Cleanup
ccall((:spir_sve_result_release, libpath), Cvoid, (Ptr{Cvoid},), sve_truncated)
ccall((:spir_sve_result_release, libpath), Cvoid, (Ptr{Cvoid},), sve)
kernel_release(kernel)

# Test 6: Basis construction
println("\n📝 Test 6: Basis Construction")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel, # kernel
    C_NULL, # sve (will compute)
    -1,     # max_size
    status_basis
)

if basis == C_NULL || status_basis[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to create basis: status = $(status_basis[])")
end

# Get basis size
basis_size = Ref{Int32}(0)
status_size = ccall(
    (:spir_basis_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, basis_size
)

if status_size != SPIR_COMPUTATION_SUCCESS
    error("Failed to get basis size: status = $status_size")
end

println("✅ Created basis")
println("   Size: $(basis_size[])")

# Get statistics
stats = Ref{Int32}(0)
status_stats = ccall(
    (:spir_basis_get_stats, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, stats
)
println("   Statistics: $(stats[] == 1 ? "Fermionic" : "Bosonic")")

# Get tau sampling points
n_taus = Ref{Int32}(0)
ccall(
    (:spir_basis_get_n_default_taus, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, n_taus
)

tau_points = Vector{Float64}(undef, n_taus[])
ccall(
    (:spir_basis_get_default_taus, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    basis, tau_points
)

println("   Tau sampling points: $(n_taus[])")
println("   First 3: $(tau_points[1:min(3, end)])")

# Get Matsubara sampling points
n_matsus = Ref{Int32}(0)
ccall(
    (:spir_basis_get_n_default_matsus, libpath),
    Int32,
    (Ptr{Cvoid}, Bool, Ref{Int32}),
    basis, true, n_matsus  # positive_only = true
)

matsu_points = Vector{Int64}(undef, n_matsus[])
ccall(
    (:spir_basis_get_default_matsus, libpath),
    Int32,
    (Ptr{Cvoid}, Bool, Ptr{Int64}),
    basis, true, matsu_points
)

println("   Matsubara sampling points (positive): $(n_matsus[])")
println("   First 3: $(matsu_points[1:min(3, end)])")

# Cleanup
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 7: Funcs API
println("\n📝 Test 7: Funcs API (u, v, uhat)")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel,
    C_NULL,
    -1,
    status_basis
)

if basis == C_NULL || status_basis[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to create basis: status = $(status_basis[])")
end

# Get u funcs (imaginary-time basis functions)
status_u = Ref{Int32}(0)
u_funcs = ccall(
    (:spir_basis_get_u, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Ref{Int32}),
    basis, status_u
)

if u_funcs == C_NULL || status_u[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to get u funcs: status = $(status_u[])")
end

println("✅ Got u funcs (imaginary-time basis functions)")

# Get v funcs (real-frequency basis functions)
status_v = Ref{Int32}(0)
v_funcs = ccall(
    (:spir_basis_get_v, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Ref{Int32}),
    basis, status_v
)

if v_funcs == C_NULL || status_v[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to get v funcs: status = $(status_v[])")
end

println("✅ Got v funcs (real-frequency basis functions)")

# Get uhat funcs (Matsubara-frequency basis functions)
status_uhat = Ref{Int32}(0)
uhat_funcs = ccall(
    (:spir_basis_get_uhat, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Ref{Int32}),
    basis, status_uhat
)

if uhat_funcs == C_NULL || status_uhat[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to get uhat funcs: status = $(status_uhat[])")
end

println("✅ Got uhat funcs (Matsubara-frequency basis functions)")

# Cleanup
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), u_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), v_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), uhat_funcs)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 8: Omega (real frequency) sampling points
println("\n📝 Test 8: Omega Sampling Points")
kernel = kernel_logistic_new(10.0)

# Create basis
status_basis = Ref{Int32}(0)
basis = ccall(
    (:spir_basis_new, libpath),
    Ptr{Cvoid},
    (Int32, Float64, Float64, Float64, Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Int32}),
    1,      # Fermionic
    10.0,   # beta
    1.0,    # omega_max
    1e-6,   # epsilon
    kernel,
    C_NULL,
    -1,
    status_basis
)

if basis == C_NULL || status_basis[] != SPIR_COMPUTATION_SUCCESS
    error("Failed to create basis: status = $(status_basis[])")
end

# Get number of omega sampling points
n_ws = Ref{Int32}(0)
status_ws = ccall(
    (:spir_basis_get_n_default_ws, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, n_ws
)

if status_ws != SPIR_COMPUTATION_SUCCESS
    error("Failed to get number of omega points: status = $status_ws")
end

println("✅ Number of omega sampling points: $(n_ws[])")

# Get omega sampling points
ws = Vector{Float64}(undef, n_ws[])
status_get_ws = ccall(
    (:spir_basis_get_default_ws, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    basis, ws
)

if status_get_ws != SPIR_COMPUTATION_SUCCESS
    error("Failed to get omega points: status = $status_get_ws")
end

println("   First 5 omega points:")
for i in 1:min(5, length(ws))
    println("     ω[$i] = $(ws[i])")
end

# Test singular_values alias
basis_size = Ref{Int32}(0)
ccall(
    (:spir_basis_get_size, libpath),
    Int32,
    (Ptr{Cvoid}, Ref{Int32}),
    basis, basis_size
)

svals_alias = Vector{Float64}(undef, basis_size[])
status_alias = ccall(
    (:spir_basis_get_singular_values, libpath),
    Int32,
    (Ptr{Cvoid}, Ptr{Float64}),
    basis, svals_alias
)

if status_alias != SPIR_COMPUTATION_SUCCESS
    error("Failed to get singular values via alias: status = $status_alias")
end

println("✅ get_singular_values (alias) works correctly")
println("   First singular value: $(svals_alias[1])")

# Cleanup
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

println("\n" * "=" ^ 50)
println("✅ All tests passed!")

