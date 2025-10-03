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

**Note**: This design document is a living document and should be updated as we gain more insights from implementation and testing.
