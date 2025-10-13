# Reduced Domain Storage Design Study

**Created**: October 13, 2025  
**Purpose**: Evaluate keeping SVE results in reduced domain [0,1]×[0,1] instead of extending to full domain [-1,1]×[-1,1]

---

## Current Implementation

### Flow
1. **Compute**: Reduced kernel K_red(x,y) for x,y ∈ [0,1]
   - Even: K(x,y) + K(x,-y)
   - Odd: K(x,y) - K(x,-y)

2. **SVD**: On reduced domain → U, S, V in [0,1]

3. **Extend**: U, V from [0,1] → [-1,1]
   - `extend_to_full_domain()` in `sve/utils.rs:74`
   - Called in `CentrosymmSVE::postprocess()` line 260-261
   - Applies 1/√2 normalization
   - Handles even/odd symmetry

4. **Merge**: Combine even and odd results
   - `merge_results()` receives extended results
   - Produces final basis functions on [-1,1]

### Current Storage
```rust
FiniteTempBasis {
    u: PiecewiseLegendrePolyVector,   // domain: [-1,1] (tau domain)
    v: PiecewiseLegendrePolyVector,   // domain: [-1,1] (omega domain)
    uhat: PiecewiseLegendreFTVector,  // Fourier transform of u
    s: Vec<f64>,                       // Singular values
    // ...
}
```

---

## Proposed Design: Reduced Domain Storage

### New Flow
1. **Compute**: Same (reduced kernel)
2. **SVD**: Same (on reduced domain)
3. **Store**: Keep U, V in [0,1] (NO extension)
4. **Evaluate**: Apply symmetry on-the-fly

### Proposed Storage

**Best Design**: Use existing `symm` field in `PiecewiseLegendrePoly`

```rust
FiniteTempBasis {
    // Reduced domain: each poly has symm field (+1=even, -1=odd)
    u_reduced: PiecewiseLegendrePolyVector,  // [0,1], symm embedded
    v_reduced: PiecewiseLegendrePolyVector,  // [0,1], symm embedded
    
    // Extended domain (lazy cache)
    u_extended: OnceCell<PiecewiseLegendrePolyVector>,  // [-1,1]
    v_extended: OnceCell<PiecewiseLegendrePolyVector>,  // [-1,1]
    
    s: Vec<f64>,  // Singular values
    // ...
}

// Each PiecewiseLegendrePoly already has:
// pub struct PiecewiseLegendrePoly {
//     pub data: DTensor<f64, 2>,
//     pub symm: i32,  // ← Parity information! +1=even, -1=odd, 0=none
//     // ...
// }
```

**No need for separate parity vector** - `symm` is already embedded in each polynomial!

**Benefits of this design**:
- ✅ **Self-contained**: Each polynomial knows its own symmetry
- ✅ **Type-safe**: Can't mix up polynomials and parities
- ✅ **Serialization-friendly**: symm saved with polynomial data
- ✅ **Matches C++/Julia**: Uses same `symm` convention
- ✅ **Single vector**: Loop over one `u_reduced` instead of two separate vectors

**Access pattern**:
```rust
// Get parity of l-th basis function
let parity = basis.u_reduced[l].symm();  // Returns +1 or -1

// Evaluate with automatic symmetry handling
pub fn evaluate_reduced(&self, l: usize, x: f64) -> f64 {
    if x >= 0.0 {
        self.u_reduced[l].evaluate(x)
    } else {
        // f(-x) = symm * f(x)
        let symm = self.u_reduced[l].symm() as f64;
        symm * self.u_reduced[l].evaluate(-x)
    }
}

// Or even simpler - add method to PiecewiseLegendrePoly:
impl PiecewiseLegendrePoly {
    pub fn evaluate_with_symmetry(&self, x: f64) -> f64 {
        if x >= self.xmin && x <= self.xmax {
            self.evaluate(x)  // Within domain
        } else if self.symm != 0 {
            // Use symmetry: f(-x) = symm * f(x)
            (self.symm as f64) * self.evaluate(-x)
        } else {
            panic!("x outside domain and no symmetry defined")
        }
    }
}
```

---

## Impact Analysis

### Benefits ✅

#### 1. Memory Efficiency
- **Coefficient storage**: ~50% reduction
  - Current: 2N segments × L basis functions
  - Proposed: N segments × L basis functions
- **Example**: Lambda=10, epsilon=1e-4
  - Current: ~30 segments × 21 basis × 2 domains = 1260 segments
  - Proposed: ~15 segments × 21 basis = 315 segments (**75% reduction!**)

#### 2. Numerical Precision
- **No 1/√2 normalization** during extension
- **Original SVD precision** preserved
- **Symmetry errors** don't accumulate

#### 3. Conceptual Clarity
- **Explicit symmetry** information preserved
- **Physical interpretation** clearer (even/odd modes)
- **Debugging** easier (separate even/odd components)

### Challenges ⚠️

#### 1. Evaluation Complexity
**Current** (simple):
```rust
fn evaluate_tau(&self, tau: Vec<f64>) -> DTensor<f64, 2> {
    // tau → x ∈ [-1,1]
    self.u[l].evaluate(x)  // Direct evaluation
}
```

**Proposed** (complex):
```rust
fn evaluate_tau(&self, tau: Vec<f64>) -> DTensor<f64, 2> {
    // tau → x ∈ [-1,1]
    if x >= 0.0 {
        self.u[l].evaluate(x)  // Positive domain
    } else {
        // Apply symmetry: f(-x) = ±f(x)
        let sign = if is_even[l] { 1.0 } else { -1.0 };
        sign * self.u[l].evaluate(-x)  // Mirror to positive domain
    }
}
```

#### 2. API Changes
- **Basis structure** changes significantly
- **Evaluation methods** need symmetry handling
- **Backward compatibility** with tests may break

#### 3. Merge Logic
- Current `merge_results()` expects extended domains
- Need new merging strategy for reduced domains
- Complexity in handling even/odd interleaving

---

## Implementation Options

### Option 1: Pure Reduced Domain (Most Aggressive)
**Store**: Only [0,1] with symmetry tags  
**Evaluate**: Apply symmetry every time  
**Memory**: 50% of current  
**Complexity**: High (all evaluation code changes)

### Option 2: Hybrid (Recommended) 🟢
**Store**: [0,1] internally, extend on first access  
**Evaluate**: Cache extended version lazily  
**Memory**: Same as current after first use, 50% before  
**Complexity**: Medium (localized changes)

```rust
FiniteTempBasis {
    // Internal: reduced domain (loaded from file/computed)
    u_reduced: OnceCell<Vec<SymmetricPoly>>,
    
    // Public API: extended domain (lazy)
    u: OnceCell<PiecewiseLegendrePolyVector>,
    
    // First call to u() triggers extension
}
```

### Option 3: Optional Extension Flag (Flexible)
**Store**: [0,1] by default  
**Extend**: User chooses via parameter  
**Memory**: User controls  
**Complexity**: Low (minimal API changes)

```rust
impl FiniteTempBasis {
    pub fn new(..., extend_domain: bool) -> Self {
        if extend_domain {
            // Current behavior
        } else {
            // Keep reduced domain
        }
    }
}
```

---

## Performance Implications

### Computation Time
| Operation | Current | Reduced (Option 1) | Reduced (Option 2) |
|-----------|---------|-------------------|-------------------|
| SVE | Same | Same | Same |
| Extension | O(N×L) once | - | O(N×L) lazy |
| Evaluate (x≥0) | O(L) | O(L) | O(L) |
| Evaluate (x<0) | O(L) | O(L) + branch | O(L) |
| **Total** | **Baseline** | **~Same** | **~Same** |

### Memory Usage
| Structure | Current | Reduced | Savings |
|-----------|---------|---------|---------|
| Coefficients | 2N×L | N×L | 50% |
| Knots | 2N | N | 50% |
| Singular values | L | L | 0% |
| **Total** | **Baseline** | **~40%** | **~40%** |

---

## Recommendation

### 🟢 **Option 2: Hybrid Approach**

**Rationale**:
1. **Best of both worlds**
   - Save memory during computation/serialization
   - Fast evaluation (extended domain cached)
   
2. **Minimal API disruption**
   - Public API unchanged (always returns extended domain)
   - Internal optimization transparent to users
   
3. **Future-proof**
   - Can add `get_reduced()` method later if needed
   - Serialization can save compact form
   - Backward compatible

**Implementation Plan**:
```rust
pub struct FiniteTempBasis<K, S> {
    kernel: K,
    beta: f64,
    
    // Reduced domain (always computed)
    u_even_reduced: PiecewiseLegendrePolyVector,  // [0,1]
    u_odd_reduced: PiecewiseLegendrePolyVector,   // [0,1]
    v_even_reduced: PiecewiseLegendrePolyVector,  // [0,1]
    v_odd_reduced: PiecewiseLegendrePolyVector,   // [0,1]
    
    // Extended domain (lazy, cached)
    u: OnceCell<PiecewiseLegendrePolyVector>,     // [-1,1]
    v: OnceCell<PiecewiseLegendrePolyVector>,     // [-1,1]
    uhat: OnceCell<PiecewiseLegendreFTVector>,
    
    s: Vec<f64>,
    _phantom: PhantomData<S>,
}

impl<K, S> FiniteTempBasis<K, S> {
    // Public API: transparent extension
    pub fn u(&self) -> &PiecewiseLegendrePolyVector {
        self.u.get_or_init(|| {
            // Extend and merge on first access
            extend_and_merge(
                &self.u_even_reduced,
                &self.u_odd_reduced,
            )
        })
    }
    
    // Advanced API: access reduced domain directly
    pub fn u_reduced(&self) -> (&PiecewiseLegendrePolyVector, &PiecewiseLegendrePolyVector) {
        (&self.u_even_reduced, &self.u_odd_reduced)
    }
}
```

---

## Next Steps

1. ✅ Document current extension logic
2. ✅ Prototype Option 2 implementation
3. ✅ Benchmark memory usage (before/after)
4. ✅ Verify all tests still pass
5. ✅ Measure evaluation performance impact

**Estimated effort**: 2-3 days for Option 2

---

## Questions to Resolve

1. **Serialization format**: Save reduced or extended?
   - **Recommendation**: Save reduced (40% smaller files)
   
2. **C-API**: Expose reduced domain?
   - **Recommendation**: Optional getter for advanced users
   
3. **Backward compatibility**: How to handle?
   - **Recommendation**: Default to extended (existing behavior)
   - Add opt-in flag for reduced domain

4. **Thread safety**: OnceCell vs Mutex?
   - **Recommendation**: `OnceCell` (initialization once, read-only after)

