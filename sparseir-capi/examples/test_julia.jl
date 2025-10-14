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

# Test 9: Function Evaluation
println("\n📝 Test 9: Function Evaluation")
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

# Get u funcs
status_u = Ref{Int32}(0)
u_funcs = ccall((:spir_basis_get_u, libpath), Ptr{Cvoid}, (Ptr{Cvoid}, Ref{Int32}), basis, status_u)

# Get size
u_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), u_funcs, u_size)

println("✅ u funcs size: $(u_size[])")

# Test single point evaluation
u_values = Vector{Float64}(undef, u_size[])
status_eval = ccall(
    (:spir_funcs_eval, libpath),
    Int32,
    (Ptr{Cvoid}, Float64, Ptr{Float64}),
    u_funcs, 0.0, u_values
)

if status_eval != SPIR_COMPUTATION_SUCCESS
    error("Failed to evaluate u: status = $status_eval")
end

println("✅ Evaluated u at tau=0")
println("   u[0](0) = $(u_values[1])")
println("   u[1](0) = $(u_values[2])")

# Test batch evaluation
tau_points = [-5.0, -2.5, 0.0, 2.5, 5.0]
u_batch = Matrix{Float64}(undef, u_size[], length(tau_points))  # column-major
status_batch = ccall(
    (:spir_funcs_batch_eval, libpath),
    Int32,
    (Ptr{Cvoid}, Int32, Int32, Ptr{Float64}, Ptr{Float64}),
    u_funcs, 1, length(tau_points), tau_points, u_batch
)

if status_batch != SPIR_COMPUTATION_SUCCESS
    error("Failed to batch evaluate u: status = $status_batch")
end

println("✅ Batch evaluated u at $(length(tau_points)) tau points")
println("   u[0](tau) = $(u_batch[1, :])")

# Get uhat funcs
status_uhat = Ref{Int32}(0)
uhat_funcs = ccall((:spir_basis_get_uhat, libpath), Ptr{Cvoid}, (Ptr{Cvoid}, Ref{Int32}), basis, status_uhat)

# Test Matsubara evaluation
uhat_values = Vector{ComplexF64}(undef, u_size[])
status_matsu = ccall(
    (:spir_funcs_eval_matsu, libpath),
    Int32,
    (Ptr{Cvoid}, Int64, Ptr{ComplexF64}),
    uhat_funcs, Int64(1), uhat_values
)

if status_matsu != SPIR_COMPUTATION_SUCCESS
    error("Failed to evaluate uhat: status = $status_matsu")
end

println("✅ Evaluated uhat at n=1 (iω_1)")
println("   uhat[0](iω_1) = $(uhat_values[1])")
println("   |uhat[0](iω_1)| = $(abs(uhat_values[1]))")

# Test batch Matsubara evaluation
matsu_ns = Int64[1, 3, 5, 7]
uhat_batch = Matrix{ComplexF64}(undef, u_size[], length(matsu_ns))
status_batch_matsu = ccall(
    (:spir_funcs_batch_eval_matsu, libpath),
    Int32,
    (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}, Ptr{ComplexF64}),
    uhat_funcs, 1, length(matsu_ns), matsu_ns, uhat_batch
)

if status_batch_matsu != SPIR_COMPUTATION_SUCCESS
    error("Failed to batch evaluate uhat: status = $status_batch_matsu")
end

println("✅ Batch evaluated uhat at $(length(matsu_ns)) Matsubara frequencies")
println("   First column magnitudes: $(abs.(uhat_batch[1:3, 1]))")

# Cleanup
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), u_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), uhat_funcs)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

# Test 10: Clone and Slice Functions
println("\n📝 Test 10: Clone and Slice Functions")
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

# Get u funcs
status_u = Ref{Int32}(0)
u_funcs = ccall((:spir_basis_get_u, libpath), Ptr{Cvoid}, (Ptr{Cvoid}, Ref{Int32}), basis, status_u)

# Test is_assigned
is_assigned = ccall((:spir_funcs_is_assigned, libpath), Int32, (Ptr{Cvoid},), u_funcs)
println("✅ is_assigned for valid object: $is_assigned")
@assert is_assigned == 1

is_null_assigned = ccall((:spir_funcs_is_assigned, libpath), Int32, (Ptr{Cvoid},), C_NULL)
println("✅ is_assigned for null: $is_null_assigned")
@assert is_null_assigned == 0

# Test clone
cloned_funcs = ccall((:spir_funcs_clone, libpath), Ptr{Cvoid}, (Ptr{Cvoid},), u_funcs)
@assert cloned_funcs != C_NULL
println("✅ Cloned funcs successfully")

# Check cloned size
u_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), u_funcs, u_size)
cloned_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), cloned_funcs, cloned_size)
@assert cloned_size[] == u_size[]
println("✅ Cloned funcs has same size ($(u_size[]))")

# Test slice
indices = Int32[0, 2, 4]  # Select first, third, fifth functions (0-indexed)
slice_status = Ref{Int32}(0)
sliced_funcs = ccall(
    (:spir_funcs_get_slice, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Int32, Ptr{Int32}, Ref{Int32}),
    u_funcs,
    length(indices),
    indices,
    slice_status
)

@assert slice_status[] == SPIR_COMPUTATION_SUCCESS
@assert sliced_funcs != C_NULL
println("✅ Created slice with $(length(indices)) functions")

# Check sliced size
sliced_size = Ref{Int32}(0)
ccall((:spir_funcs_get_size, libpath), Int32, (Ptr{Cvoid}, Ref{Int32}), sliced_funcs, sliced_size)
@assert sliced_size[] == length(indices)
println("✅ Sliced funcs has correct size ($(sliced_size[]))")

# Evaluate sliced functions
sliced_values = Vector{Float64}(undef, sliced_size[])
status_eval = ccall(
    (:spir_funcs_eval, libpath),
    Int32,
    (Ptr{Cvoid}, Float64, Ptr{Float64}),
    sliced_funcs, 0.0, sliced_values
)
@assert status_eval == SPIR_COMPUTATION_SUCCESS
println("✅ Sliced funcs evaluates correctly")
println("   Values: $(sliced_values)")

# Test error case: negative index
bad_indices = Int32[-1]
bad_status = Ref{Int32}(0)
bad_slice = ccall(
    (:spir_funcs_get_slice, libpath),
    Ptr{Cvoid},
    (Ptr{Cvoid}, Int32, Ptr{Int32}, Ref{Int32}),
    u_funcs,
    length(bad_indices),
    bad_indices,
    bad_status
)
@assert bad_status[] == SPIR_INVALID_ARGUMENT
@assert bad_slice == C_NULL
println("✅ Negative index correctly rejected")

# Cleanup
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), sliced_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), cloned_funcs)
ccall((:spir_funcs_release, libpath), Cvoid, (Ptr{Cvoid},), u_funcs)
ccall((:spir_basis_release, libpath), Cvoid, (Ptr{Cvoid},), basis)
kernel_release(kernel)

println("\n" * "=" ^ 50)
println("✅ All tests passed!")

