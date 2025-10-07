# SparseIR Rust Design Document

## Poly Module Design

### Overview

The `poly` module implements piecewise Legendre polynomial functionality, which is a core component of SparseIR. This module needs to carefully handle memory management, ownership, and C-API compatibility.

### Current C-API Analysis

From the C-API specification, we have:

1. **`PiecewiseLegendrePoly`** - Immutable polynomial object
2. **`PiecewiseLegendrePolyVector`** - Container for multiple polynomials
3. **Extraction capability** - Individual polynomials can be extracted from the vector as `PiecewiseLegendrePoly` objects

### Key Design Questions

#### 1. Ownership Model

**Question**: How should we handle ownership between `PiecewiseLegendrePolyVector` and individual `PiecewiseLegendrePoly` objects?

**Options**:
- **Option A**: Copy-on-extraction (each extraction creates a new owned polynomial)
- **Option B**: Shared ownership with `Arc<Rc<...>>` or similar
- **Option C**: Borrowing with lifetime parameters
- **Option D**: Hybrid approach with both owned and borrowed variants

**C-API Constraints**:
- `PiecewiseLegendrePoly` objects are immutable
- Individual polynomials can be extracted from vectors
- No clear ownership semantics in C API

#### 2. Memory Layout

**Question**: How should we represent the internal data structure?

**Considerations**:
- **Contiguous storage**: Store all polynomial data in a single allocation
- **Individual storage**: Each polynomial manages its own memory
- **Hybrid**: Vector owns the storage, polynomials reference into it

#### 3. Rust vs C-API Mapping

**Question**: How do we map Rust ownership to C-API semantics?

**Challenges**:
- C-API doesn't specify ownership clearly
- Rust requires explicit ownership decisions
- Need to support both owned and borrowed access patterns

### Proposed Design Options

#### Option 1: Copy-Based Design

```rust
pub struct PiecewiseLegendrePoly {
    // Owned data
    coefficients: Vec<f64>,
    intervals: Vec<(f64, f64)>,
    degree: usize,
}

pub struct PiecewiseLegendrePolyVector {
    polynomials: Vec<PiecewiseLegendrePoly>,  // Each polynomial is owned
}

impl PiecewiseLegendrePolyVector {
    pub fn get(&self, index: usize) -> Option<&PiecewiseLegendrePoly> {
        self.polynomials.get(index)
    }
    
    pub fn extract(&self, index: usize) -> Option<PiecewiseLegendrePoly> {
        self.polynomials.get(index).cloned()  // Copy on extraction
    }
}
```

**Pros**:
- Simple ownership model
- Clear memory management
- No lifetime issues

**Cons**:
- Memory overhead for copies
- May not match C-API performance expectations

#### Option 2: Shared Ownership Design

```rust
use std::sync::Arc;

pub struct PiecewiseLegendrePoly {
    data: Arc<PolynomialData>,
    index: usize,  // Index within the shared data
}

pub struct PiecewiseLegendrePolyVector {
    shared_data: Arc<PolynomialData>,
    polynomial_count: usize,
}

struct PolynomialData {
    coefficients: Vec<f64>,
    intervals: Vec<(f64, f64)>,
    degrees: Vec<usize>,
}

impl PiecewiseLegendrePolyVector {
    pub fn extract(&self, index: usize) -> Option<PiecewiseLegendrePoly> {
        if index < self.polynomial_count {
            Some(PiecewiseLegendrePoly {
                data: Arc::clone(&self.shared_data),
                index,
            })
        } else {
            None
        }
    }
}
```

**Pros**:
- Memory efficient (no copying)
- Matches C-API performance expectations
- Clear ownership semantics

**Cons**:
- More complex implementation
- `Arc` overhead (though minimal)

#### Option 3: Borrowing Design

```rust
pub struct PiecewiseLegendrePoly<'a> {
    coefficients: &'a [f64],
    intervals: &'a [(f64, f64)],
    degree: usize,
}

pub struct PiecewiseLegendrePolyVector {
    coefficients: Vec<f64>,
    intervals: Vec<(f64, f64)>,
    degrees: Vec<usize>,
    offsets: Vec<usize>,  // Start indices for each polynomial
}

impl PiecewiseLegendrePolyVector {
    pub fn get(&self, index: usize) -> Option<PiecewiseLegendrePoly<'_>> {
        // Return borrowed polynomial
    }
    
    pub fn extract(&self, index: usize) -> Option<PiecewiseLegendrePoly<'_>> {
        // Same as get - lifetime tied to vector
    }
}
```

**Pros**:
- Zero-copy access
- Lifetime safety guaranteed by Rust
- Most memory efficient

**Cons**:
- Lifetime complexity
- May not match C-API expectations
- Borrowing limitations

#### Option 4: Hybrid Design

```rust
pub enum PiecewiseLegendrePoly {
    Owned {
        coefficients: Vec<f64>,
        intervals: Vec<(f64, f64)>,
        degree: usize,
    },
    Borrowed {
        coefficients: &'static [f64],
        intervals: &'static [(f64, f64)],
        degree: usize,
    },
    Shared {
        data: Arc<PolynomialData>,
        index: usize,
    },
}

pub struct PiecewiseLegendrePolyVector {
    polynomials: Vec<PiecewiseLegendrePoly>,
}
```

**Pros**:
- Maximum flexibility
- Can optimize for different use cases

**Cons**:
- Complex implementation
- May be over-engineered

### C-API Compatibility Considerations

#### Immutable Access Pattern

The C-API suggests that `PiecewiseLegendrePoly` objects are immutable. This aligns well with Rust's ownership model.

#### Extraction Semantics

The C-API allows extracting individual polynomials from vectors. The key question is whether this extraction:
1. Transfers ownership (C-API caller owns the result)
2. Creates a copy (C-API caller gets a copy)
3. Creates a reference (lifetime tied to the vector)

### Performance Considerations

#### Memory Access Patterns

- **Contiguous access**: Better for cache performance
- **Individual allocation**: More flexible but potentially slower
- **Shared ownership**: Good balance of performance and flexibility

#### Copy vs Reference Trade-offs

- **Copy**: Simple but memory-intensive
- **Reference**: Memory-efficient but lifetime-complex
- **Shared ownership**: Good middle ground

### Recommended Approach

Based on the analysis, I recommend **Option 2: Shared Ownership Design with Integrated Data Model** for the following reasons:

1. **Memory Efficiency**: No unnecessary copying, unified data storage
2. **Cache Performance**: Contiguous memory access for better cache locality
3. **C-API Compatibility**: Matches expected performance characteristics
4. **Ownership Clarity**: Clear ownership semantics
5. **Rust Idioms**: Uses standard Rust patterns (`Arc`)
6. **Extensibility**: Easy to add new features

### Final Design: Integrated Data Model

The recommended design uses a unified data structure where all polynomial data is stored in a single `PolynomialData` structure, shared across multiple `PiecewiseLegendrePoly` objects.

#### Data Structure

```rust
pub struct PiecewiseLegendrePolyVector {
    shared_data: Arc<PolynomialData>,  // Unified data storage
    polynomials: Vec<PiecewiseLegendrePoly>,  // Array of polynomial references
}

pub struct PiecewiseLegendrePoly {
    data: Arc<PolynomialData>,  // Shared reference to unified data
    index: usize,               // Index within the unified data
}

struct PolynomialData {
    // Unified coefficient storage
    all_coefficients: Vec<f64>,
    
    // Unified interval storage
    all_intervals: Vec<(f64, f64)>,
    
    // Metadata for each polynomial
    polynomial_info: Vec<PolynomialInfo>,
}

struct PolynomialInfo {
    coefficient_start: usize,  // Start index in all_coefficients
    coefficient_count: usize,  // Number of coefficients
    interval_start: usize,     // Start index in all_intervals
    interval_count: usize,     // Number of intervals
    degree: usize,            // Polynomial degree
}
```

#### Key Benefits

1. **Memory Efficiency**: Single large allocation instead of multiple small ones
2. **Cache Performance**: Contiguous memory access pattern
3. **Low Extraction Cost**: Only `Arc::clone()` overhead (few nanoseconds)
4. **Immutable Design**: Both vector and individual polynomials are immutable
5. **C-API Compatibility**: Array-based access pattern similar to C++ implementation

#### Performance Characteristics

| Operation | Integrated Data | Individual Data |
|-----------|----------------|-----------------|
| **Creation** | Fast (single allocation) | Slow (multiple allocations) |
| **Access** | Fast (cache efficient) | Slow (memory scattered) |
| **Extraction** | Minimal (Arc::clone) | Minimal (Arc::clone) |
| **Memory Usage** | Low | High |

#### Usage Pattern

```rust
// Create vector with unified data
let vector = PiecewiseLegendrePolyVector::new(polynomials);

// Direct access (no copy)
let poly = vector.get(0);

// Extract (minimal copy cost)
let poly_copy = vector.extract(0);

// Iteration
for poly in &vector.polynomials {
    let value = poly.evaluate(x);
}
```

### Implementation Plan

#### Phase 1: Core Data Structures
```rust
pub struct PiecewiseLegendrePolyVector {
    shared_data: Arc<PolynomialData>,
    polynomials: Vec<PiecewiseLegendrePoly>,
}

pub struct PiecewiseLegendrePoly {
    data: Arc<PolynomialData>,
    index: usize,
}

struct PolynomialData {
    all_coefficients: Vec<f64>,
    all_intervals: Vec<(f64, f64)>,
    polynomial_info: Vec<PolynomialInfo>,
}
```

#### Phase 2: Basic Operations
- Polynomial evaluation
- Vector operations
- Extraction methods

#### Phase 3: C-API Integration
- FFI bindings
- Memory management
- Error handling

### Open Questions

1. **Degree Storage**: Should degree be stored per polynomial or computed on-demand?
2. **Interval Representation**: How should we represent piecewise intervals?
3. **Coefficient Storage**: Flat array vs. nested structure?
4. **Error Handling**: How should we handle invalid polynomial data?
5. **Serialization**: Do we need serialization support?

### Next Steps

1. **Feedback Collection**: Gather input on the proposed design
2. **Prototype Implementation**: Create a minimal working version
3. **Benchmarking**: Compare performance with C++ implementation
4. **C-API Integration**: Implement FFI layer
5. **Documentation**: Create comprehensive documentation

---

## SVE (Singular Value Expansion) Module Design

### Overview

The `sve` module implements Singular Value Expansion computation for integral kernels. This is the core algorithm of SparseIR that decomposes kernels into singular functions and values.

### Design Principles

1. **Separation of Concerns**: Clear separation between general SVE processing and symmetry-specific logic
2. **Type-driven Precision**: Automatic selection of working precision based on required accuracy
3. **Symmetry Exploitation**: Efficient computation for centrosymmetric kernels via even/odd decomposition
4. **Composability**: Modular design allowing easy extension and testing

### Module Structure

```
sve/
├── mod.rs           # Module exports
├── result.rs        # SVEResult definition
├── strategy.rs      # SVEStrategy trait, CentrosymmSVE, SamplingSVE
├── compute.rs       # Main computation functions (compute_sve, truncate, compute_svd)
├── types.rs         # TworkType, SVDStrategy, safe_epsilon
└── utils.rs         # Helper functions (extend_to_full_domain, etc.)
```

### Key Components

#### 1. SVEResult (`result.rs`)

Container for SVE computation results.

```rust
pub struct SVEResult {
    pub u: PiecewiseLegendrePolyVector,  // Left singular functions
    pub s: Array1<f64>,                  // Singular values (decreasing order)
    pub v: PiecewiseLegendrePolyVector,  // Right singular functions
    pub epsilon: f64,                    // Accuracy parameter
}
```

**Methods**:
- `new()`: Constructor
- `part()`: Extract subset based on threshold

#### 2. Working Precision Types (`types.rs`)

```rust
pub enum TworkType {
    Float64 = 0,      // Double precision (64-bit)
    Float64X2 = 1,    // Extended precision (128-bit double-double)
    Auto = -1,        // Automatic selection based on epsilon
}

pub enum SVDStrategy {
    Fast = 0,         // Fast computation
    Accurate = 1,     // Accurate computation
    Auto = -1,        // Automatic selection
}
```

**safe_epsilon function**:

Determines safe achievable epsilon and working precision:

```rust
pub fn safe_epsilon(
    epsilon: f64,
    twork: TworkType,
    svd_strategy: SVDStrategy,
) -> (f64, TworkType, SVDStrategy) {
    // Check for negative epsilon
    if epsilon < 0.0 {
        panic!("eps_required must be non-negative");
    }
    
    // Choose working precision
    let twork_actual = match twork {
        TworkType::Auto => {
            if epsilon.is_nan() || epsilon < 1e-8 {
                TworkType::Float64X2  // High accuracy needed
            } else {
                TworkType::Float64
            }
        }
        other => other,
    };
    
    // Determine safe epsilon for chosen precision
    let safe_eps = match twork_actual {
        TworkType::Float64 => 1e-8,      // ~1.5e-8 technically
        TworkType::Float64X2 => 1e-15,   // Double-double precision
        _ => 1e-8,
    };
    
    // Choose SVD strategy
    let svd_strategy_actual = match svd_strategy {
        SVDStrategy::Auto => {
            if !epsilon.is_nan() && epsilon < safe_eps {
                SVDStrategy::Accurate  // User wants higher precision
            } else {
                SVDStrategy::Fast
            }
        }
        other => other,
    };
    
    (safe_eps, twork_actual, svd_strategy_actual)
}
```

#### 3. SVE Strategy Trait (`strategy.rs`)

```rust
pub trait SVEStrategy<T: CustomNumeric> {
    /// Compute discretized matrices for SVD
    fn matrices(&self) -> Vec<Array2<T>>;
    
    /// Post-process SVD results to create SVEResult
    fn postprocess(
        &self,
        u_list: Vec<Array2<T>>,
        s_list: Vec<Array1<T>>,
        v_list: Vec<Array2<T>>,
    ) -> SVEResult;
}
```

#### 4. SamplingSVE - General SVE Processor (`strategy.rs`)

**Responsibility**: General SVE processing without symmetry knowledge.

**Key Characteristic**: Does NOT know about kernel symmetry - only processes discretized matrices.

```rust
pub struct SamplingSVE<T>
where
    T: CustomNumeric + Send + Sync + 'static,
{
    segments_x: Vec<T>,      // Segment boundaries for x
    segments_y: Vec<T>,      // Segment boundaries for y
    gauss_x: Rule<T>,        // Gauss quadrature for x
    gauss_y: Rule<T>,        // Gauss quadrature for y
    epsilon: f64,
    n_gauss: usize,
}
```

**Methods**:
- `new()`: Create from geometric information (segments, Gauss rules)
- `postprocess_single()`: Convert single SVD result to polynomials

**Key Design Decision**: 
- Takes only geometric information (segments, Gauss rules)
- Does NOT take kernel or symmetry information
- Produces polynomials on the domain specified by segments (e.g., [0, xmax] for reduced kernels)
- Domain extension is caller's responsibility

#### 5. CentrosymmSVE - Symmetry Manager (`strategy.rs`)

**Responsibility**: Manage centrosymmetric kernel computation with even/odd decomposition.

```rust
pub struct CentrosymmSVE<T, K>
where
    T: CustomNumeric + Send + Sync + 'static,
    K: CentrosymmKernel + KernelProperties,
{
    kernel: K,
    epsilon: f64,
    hints: K::SVEHintsType<T>,
    n_gauss: usize,
    
    // Geometric information (positive domain [0, xmax])
    segments_x: Vec<T>,
    segments_y: Vec<T>,
    gauss_x: Rule<T>,
    gauss_y: Rule<T>,
    
    // General SVE processor (no symmetry knowledge)
    sampling_sve: SamplingSVE<T>,
}
```

**Flow**:
1. `matrices()`: Create reduced kernel matrices for even/odd symmetries
2. (External: SVD computation)
3. `postprocess()`:
   - Use `SamplingSVE` to convert SVD → polynomials on [0, xmax]
   - Extend to full domain [-xmax, xmax] using symmetry
   - Merge even/odd results

**Symmetry Management**:

```rust
impl CentrosymmSVE {
    fn compute_reduced_matrix(&self, symmetry: SymmetryType) -> Array2<T> {
        // Compute K_red(x, y) = K(x, y) + sign * K(x, -y)
        // where sign = +1 for Even, -1 for Odd
        let discretized = matrix_from_gauss_with_segments(
            &self.kernel,
            &self.gauss_x,
            &self.gauss_y,
            symmetry,
            &self.hints,
        );
        discretized.apply_weights_for_sve()
    }
    
    fn extend_result_to_full_domain(
        &self,
        result: (PiecewiseLegendrePolyVector, Array1<f64>, PiecewiseLegendrePolyVector),
        symmetry: SymmetryType,
    ) -> (...) {
        let (u, s, v) = result;
        
        // Extend from [0, xmax] to [-xmax, xmax]
        let u_full = extend_to_full_domain(
            u.get_polys().to_vec(),
            symmetry,
            self.kernel.xmax(),
        );
        // Same for v...
    }
}
```

#### 6. Main Computation Flow (`compute.rs`)

```rust
pub fn compute_sve<K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
) -> SVEResult
where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    // 1. Determine safe epsilon and working precision
    let (safe_epsilon, twork_actual, _) = 
        safe_epsilon(epsilon, twork, SVDStrategy::Auto);
    
    // 2. Dispatch based on working precision
    match twork_actual {
        TworkType::Float64 => {
            compute_sve_with_precision::<f64, K>(
                kernel, safe_epsilon, cutoff, max_num_svals
            )
        }
        TworkType::Float64X2 => {
            compute_sve_with_precision::<twofloat::TwoFloat, K>(
                kernel, safe_epsilon, cutoff, max_num_svals
            )
        }
        _ => panic!("Invalid TworkType"),
    }
}

fn compute_sve_with_precision<T, K>(...) -> SVEResult {
    // 1. Determine strategy (automatically chooses CentrosymmSVE)
    let sve = determine_sve::<T, K>(kernel, epsilon);
    
    // 2. Compute matrices
    let matrices = sve.matrices();
    
    // 3. Compute SVD for each matrix
    let (u_list, s_list, v_list) = compute_svd_list(&matrices);
    
    // 4. Truncate based on cutoff
    let (u_trunc, s_trunc, v_trunc) = 
        truncate(u_list, s_list, v_list, rtol, max_num_svals);
    
    // 5. Post-process to create SVEResult
    sve.postprocess(u_trunc, s_trunc, v_trunc)
}
```

### Responsibility Separation

```
┌─────────────────────────────────────────────────────────────────┐
│ CentrosymmSVE (Symmetry Management)                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. matrices():                                                  │
│    - compute_reduced_matrix(Even)  → K_red_even on [0, xmax]   │
│    - compute_reduced_matrix(Odd)   → K_red_odd on [0, xmax]    │
│                                                                  │
│ 2. postprocess():                                               │
│    ┌─────────────────────────────────────────────────────┐     │
│    │ SamplingSVE (General Processing - no symmetry)     │     │
│    ├─────────────────────────────────────────────────────┤     │
│    │ - Remove weights from SVD results                   │     │
│    │ - Convert to polynomials on [0, xmax]              │     │
│    └─────────────────────────────────────────────────────┘     │
│    ↓                                                             │
│    - extend_to_full_domain(Even) → polynomials on [-xmax, xmax]│
│    - extend_to_full_domain(Odd)  → polynomials on [-xmax, xmax]│
│    ↓                                                             │
│    - merge_results() → Final SVEResult                          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **SamplingSVE is symmetry-agnostic**:
   - Only knows about geometric discretization
   - Produces polynomials on whatever domain specified by segments
   - Domain extension is caller's responsibility

2. **CentrosymmSVE manages symmetry**:
   - Creates reduced kernels with symmetry
   - Uses SamplingSVE for general processing
   - Handles domain extension with symmetry rules
   - Merges even/odd results

3. **Clear separation of concerns**:
   - `compute.rs`: Orchestration and precision dispatch
   - `strategy.rs`: Strategy pattern for different kernel types
   - `SamplingSVE`: General SVE processing
   - `CentrosymmSVE`: Symmetry-specific logic
   - `utils.rs`: Reusable helper functions

4. **Type-driven precision**:
   - `safe_epsilon()` determines working precision
   - Generic over numeric type `T`
   - Automatic selection based on required accuracy

### Helper Functions (`utils.rs`)

```rust
/// Remove Gauss weights from SVD matrix
pub fn remove_weights<T: CustomNumeric>(
    matrix: &Array2<T>,
    weights: &[T],
    is_row: bool,
) -> Array2<T>

/// Convert SVD matrix to piecewise Legendre polynomials
pub fn svd_to_polynomials<T: CustomNumeric>(
    u_or_v: &Array2<T>,
    segments: &[T],
    gauss_rule: &Rule<f64>,
    n_gauss: usize,
) -> Vec<PiecewiseLegendrePoly>

/// Extend polynomials from [0, xmax] to [-xmax, xmax]
pub fn extend_to_full_domain(
    polys: Vec<PiecewiseLegendrePoly>,
    symmetry: SymmetryType,
    xmax: f64,
) -> Vec<PiecewiseLegendrePoly>

/// Create Legendre collocation matrix
fn legendre_collocation(
    gauss_rule: &Rule<f64>,
    n_gauss: usize,
) -> Array2<f64>

/// Merge even and odd SVE results
fn merge_results(
    result_even: (...),
    result_odd: (...),
    epsilon: f64,
) -> SVEResult
```

### Usage Example

```rust
use sparseir_rust::{compute_sve, LogisticKernel, TworkType};

fn main() {
    let lambda = 100.0;
    let epsilon = 1e-10;
    
    let kernel = LogisticKernel::new(lambda);
    
    // Automatic precision selection and CentrosymmSVE strategy
    let result = compute_sve(
        kernel,
        epsilon,
        None,  // cutoff
        None,  // max_num_svals
        TworkType::Auto,  // Automatic precision selection
    );
    
    println!("Computed {} singular values", result.s.len());
    println!("Largest singular value: {}", result.s[0]);
    
    // Extract high-accuracy subset
    let (u_part, s_part, v_part) = result.part(Some(1e-12), Some(50));
}
```

### Testing Strategy

1. **Unit Tests**:
   - `safe_epsilon` logic
   - `truncate` functionality
   - `remove_weights` correctness
   - `extend_to_full_domain` symmetry properties

2. **Integration Tests**:
   - Full SVE computation with LogisticKernel
   - Comparison with reference implementation
   - Precision tests (f64 vs TwoFloat)

3. **Property Tests**:
   - Orthogonality of singular functions
   - Singular value ordering
   - Symmetry preservation

### Open Questions

1. **Legendre Collocation Matrix**: Implementation details needed
2. **Performance Optimization**: BLAS/LAPACK integration for matrix operations
3. **Error Handling**: Graceful degradation vs panic for numerical issues
4. **Parallelization**: Thread-safe SVD computation for multiple matrices

### Future Enhancements

1. **Non-centrosymmetric kernels**: Extend to general kernels without symmetry
2. **Adaptive precision**: Dynamic precision adjustment based on convergence
3. **Caching**: Cache intermediate results for repeated computations
4. **Streaming SVD**: Process large matrices in chunks

---

**Note**: This design document is a living document and should be updated as we gain more insights from implementation and testing.
