//! Piecewise Legendre polynomial Fourier transform implementation for SparseIR
//!
//! This module provides Fourier transform functionality for piecewise Legendre
//! polynomials, enabling evaluation in Matsubara frequency domain.

use num_complex::Complex64;
use ndarray;
use std::f64::consts::PI;

use crate::traits::{StatisticsType, Fermionic, Bosonic, Statistics};
use crate::freq::{MatsubaraFreq, FermionicFreq, BosonicFreq};
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};

/// Power model for asymptotic behavior
#[derive(Debug, Clone)]
pub struct PowerModel {
    pub moments: Vec<f64>,
}

impl PowerModel {
    /// Create a new power model with given moments
    pub fn new(moments: Vec<f64>) -> Self {
        Self { moments }
    }
}

/// Piecewise Legendre polynomial with Fourier transform capability
/// 
/// This represents a piecewise Legendre polynomial that can be evaluated
/// in the Matsubara frequency domain using Fourier transform.
#[derive(Debug, Clone)]
pub struct PiecewiseLegendreFT<S: StatisticsType> {
    /// The underlying piecewise Legendre polynomial
    pub poly: PiecewiseLegendrePoly,
    /// Asymptotic cutoff frequency index
    pub n_asymp: f64,
    /// Power model for asymptotic behavior
    pub model: PowerModel,
    _phantom: std::marker::PhantomData<S>,
}

// Type aliases for convenience
pub type FermionicPiecewiseLegendreFT = PiecewiseLegendreFT<Fermionic>;
pub type BosonicPiecewiseLegendreFT = PiecewiseLegendreFT<Bosonic>;

impl<S: StatisticsType> PiecewiseLegendreFT<S> {
    /// Create a new PiecewiseLegendreFT from a polynomial and statistics
    /// 
    /// # Arguments
    /// * `poly` - The underlying piecewise Legendre polynomial
    /// * `stat` - Statistics type (Fermionic or Bosonic)
    /// * `n_asymp` - Asymptotic cutoff frequency index (default: infinity)
    /// 
    /// # Panics
    /// Panics if the polynomial domain is not [-1, 1]
    pub fn new(
        poly: PiecewiseLegendrePoly,
        _stat: S,
        n_asymp: Option<f64>,
    ) -> Self {
        // Validate domain
        if (poly.xmin - (-1.0)).abs() > 1e-12 || (poly.xmax - 1.0).abs() > 1e-12 {
            panic!("Only interval [-1, 1] is supported for Fourier transform");
        }
        
        let n_asymp = n_asymp.unwrap_or(f64::INFINITY);
        let model = Self::power_model(&poly);
        
        Self {
            poly,
            n_asymp,
            model,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the asymptotic cutoff frequency index
    pub fn get_n_asymp(&self) -> f64 {
        self.n_asymp
    }
    
    /// Get the statistics type
    pub fn get_statistics(&self) -> Statistics {
        S::STATISTICS
    }
    
    /// Get the zeta value for this statistics type
    pub fn zeta(&self) -> i64 {
        match S::STATISTICS {
            Statistics::Fermionic => 1,
            Statistics::Bosonic => 0,
        }
    }
    
    /// Get a reference to the underlying polynomial
    pub fn get_poly(&self) -> &PiecewiseLegendrePoly {
        &self.poly
    }
    
    /// Evaluate the Fourier transform at a Matsubara frequency
    /// 
    /// # Arguments
    /// * `omega` - Matsubara frequency
    /// 
    /// # Returns
    /// The complex Fourier transform value
    pub fn evaluate(&self, omega: &MatsubaraFreq<S>) -> Complex64 {
        let n = omega.get_n() as i32;
        if (n as f64).abs() < self.n_asymp {
            self.compute_unl_inner(&self.poly, n)
        } else {
            self.giw(n)
        }
    }
    
    /// Evaluate at integer Matsubara index
    pub fn evaluate_at_n(&self, n: i64) -> Complex64 {
        match MatsubaraFreq::<S>::new(n) {
            Ok(omega) => self.evaluate(&omega),
            Err(_) => Complex64::new(0.0, 0.0), // Return zero for invalid frequencies
        }
    }
    
    /// Evaluate at multiple Matsubara indices
    pub fn evaluate_at_ns(&self, ns: &[i64]) -> Vec<Complex64> {
        ns.iter().map(|&n| self.evaluate_at_n(n)).collect()
    }
    
    /// Create power model for asymptotic behavior
    fn power_model(poly: &PiecewiseLegendrePoly) -> PowerModel {
        let deriv_x1 = poly.derivs(1.0);
        let moments = Self::power_moments(&deriv_x1, poly.l);
        PowerModel::new(moments)
    }
    
    /// Compute power moments for asymptotic expansion
    fn power_moments(deriv_x1: &[f64], l: i32) -> Vec<f64> {
        let statsign = match S::STATISTICS {
            Statistics::Fermionic => -1.0,
            Statistics::Bosonic => 1.0,
        };
        
        let mut moments = deriv_x1.to_vec();
        for (m, moment) in moments.iter_mut().enumerate() {
            let m_f64 = (m + 1) as f64; // Julia uses 1-based indexing
            *moment *= -(statsign * (-1.0_f64).powi(m_f64 as i32) + (-1.0_f64).powi(l)) / 2.0_f64.sqrt();
        }
        moments
    }
    
    /// Compute the inner Fourier transform (for small frequencies)
    fn compute_unl_inner(&self, poly: &PiecewiseLegendrePoly, wn: i32) -> Complex64 {
        let wred = PI / 4.0 * wn as f64;
        let phase_wi = Self::phase_stable(poly, wn);
        let mut res = Complex64::new(0.0, 0.0);
        
        let order_max = poly.polyorder;
        let segment_count = poly.knots.len() - 1;
        
        for order in 0..order_max {
            for j in 0..segment_count {
                let data_oj = poly.data[[order, j]];
                let tnl = Self::get_tnl(order as i32, wred * poly.delta_x[j]);
                res += data_oj * tnl * phase_wi[j] / poly.norms[j];
            }
        }
        
        res / 2.0_f64.sqrt()
    }
    
    /// Compute asymptotic behavior (for large frequencies)
    fn giw(&self, wn: i32) -> Complex64 {
        let iw = Complex64::new(0.0, PI / 2.0 * wn as f64);
        if wn == 0 {
            return Complex64::new(0.0, 0.0);
        }
        
        let inv_iw = 1.0 / iw;
        let result = inv_iw * Self::evalpoly(inv_iw, &self.model.moments);
        result
    }
    
    /// Evaluate polynomial at complex point (Horner's method)
    fn evalpoly(x: Complex64, coeffs: &[f64]) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        for i in (0..coeffs.len()).rev() {
            result = result * x + Complex64::new(coeffs[i], 0.0);
        }
        result
    }
    
    /// Compute stable phase factors
    fn phase_stable(poly: &PiecewiseLegendrePoly, wn: i32) -> Vec<Complex64> {
        let mut phase_wi = Vec::with_capacity(poly.knots.len() - 1);
        let pi_over_4 = PI / 4.0;
        
        for j in 0..poly.knots.len() - 1 {
            let xm = poly.xm[j];
            let phase = Complex64::new(
                (pi_over_4 * wn as f64 * xm).cos(),
                (pi_over_4 * wn as f64 * xm).sin(),
            );
            phase_wi.push(phase);
        }
        
        phase_wi
    }
    
    /// Find sign changes in the Fourier transform
    /// 
    /// # Arguments
    /// * `positive_only` - If true, only return positive frequency sign changes
    /// 
    /// # Returns
    /// Vector of Matsubara frequencies where sign changes occur
    pub fn sign_changes(&self, positive_only: bool) -> Vec<MatsubaraFreq<S>> {
        let grid = Self::default_grid();
        let f = Self::func_for_part(self);
        let x0 = Self::find_all_roots(&f, &grid);
        
        // Transform grid indices to Matsubara frequencies
        let mut matsubara_indices: Vec<i64> = x0.into_iter()
            .map(|x| 2 * x + self.zeta())
            .collect();
        
        if !positive_only {
            Self::symmetrize_matsubara_inplace(&mut matsubara_indices);
        }
        
        matsubara_indices.into_iter()
            .filter_map(|n| MatsubaraFreq::<S>::new(n).ok())
            .collect()
    }
    
    /// Find extrema in the Fourier transform
    /// 
    /// # Arguments
    /// * `positive_only` - If true, only return positive frequency extrema
    /// 
    /// # Returns
    /// Vector of Matsubara frequencies where extrema occur
    pub fn find_extrema(&self, positive_only: bool) -> Vec<MatsubaraFreq<S>> {
        let grid = Self::default_grid();
        let f = Self::func_for_part(self);
        let x0 = Self::discrete_extrema(&f, &grid);
        
        // Transform grid indices to Matsubara frequencies
        let mut matsubara_indices: Vec<i64> = x0.into_iter()
            .map(|x| 2 * x + self.zeta())
            .collect();
        
        if !positive_only {
            Self::symmetrize_matsubara_inplace(&mut matsubara_indices);
        }
        
        matsubara_indices.into_iter()
            .filter_map(|n| MatsubaraFreq::<S>::new(n).ok())
            .collect()
    }
    
    /// Create function for extracting real or imaginary part based on parity
    fn func_for_part(&self) -> impl Fn(i64) -> f64 + '_ {
        let parity = self.poly.symm;
        let poly_ft = self.clone();
        
        move |n| {
            let omega = match MatsubaraFreq::<S>::new(n) {
                Ok(omega) => omega,
                Err(_) => return 0.0,
            };
            let value = poly_ft.evaluate(&omega);
            
            match parity {
                1 => {
                    match S::STATISTICS {
                        Statistics::Fermionic => value.im,
                        Statistics::Bosonic => value.re,
                    }
                },
                -1 => {
                    match S::STATISTICS {
                        Statistics::Fermionic => value.re,
                        Statistics::Bosonic => value.im,
                    }
                },
                _ => panic!("Cannot detect parity for symm = {}", parity),
            }
        }
    }
    
    /// Default grid for sign change detection (same as C++ DEFAULT_GRID)
    fn default_grid() -> Vec<i64> {
        // This should match the C++ DEFAULT_GRID
        // For now, use a reasonable range
        (-1000..=1000).collect()
    }
    
    /// Find all roots using the same algorithm as the poly module
    fn find_all_roots<F>(f: &F, xgrid: &[i64]) -> Vec<i64> 
    where 
        F: Fn(i64) -> f64,
    {
        if xgrid.is_empty() {
            return Vec::new();
        }
        
        // Evaluate function at all grid points
        let fx: Vec<f64> = xgrid.iter().map(|&x| f(x)).collect();
        
        // Find exact zeros (direct hits)
        let mut x_hit = Vec::new();
        for i in 0..fx.len() {
            if fx[i] == 0.0 {
                x_hit.push(xgrid[i]);
            }
        }
        
        // Find sign changes
        let mut sign_change = Vec::new();
        for i in 0..fx.len() - 1 {
            let has_sign_change = fx[i].signum() != fx[i + 1].signum();
            let not_hit = fx[i] != 0.0 && fx[i + 1] != 0.0;
            sign_change.push(has_sign_change && not_hit);
        }
        
        // If no sign changes, return only direct hits
        if sign_change.iter().all(|&sc| !sc) {
            x_hit.sort();
            return x_hit;
        }
        
        // Find intervals with sign changes
        let mut a_intervals = Vec::new();
        let mut b_intervals = Vec::new();
        let mut fa_values = Vec::new();
        
        for i in 0..sign_change.len() {
            if sign_change[i] {
                a_intervals.push(xgrid[i]);
                b_intervals.push(xgrid[i + 1]);
                fa_values.push(fx[i]);
            }
        }
        
        // Use bisection for each interval with sign change
        for i in 0..a_intervals.len() {
            let root = Self::bisect(&f, a_intervals[i], b_intervals[i], fa_values[i]);
            x_hit.push(root);
        }
        
        // Sort and return
        x_hit.sort();
        x_hit
    }
    
    /// Bisection method for integer grid
    fn bisect<F>(f: &F, a: i64, b: i64, fa: f64) -> i64 
    where 
        F: Fn(i64) -> f64,
    {
        let mut a = a;
        let mut b = b;
        let mut fa = fa;
        
        loop {
            if (b - a).abs() <= 1 {
                return a;
            }
            
            let mid = (a + b) / 2;
            let fmid = f(mid);
            
            if fa.signum() != fmid.signum() {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
    }
    
    /// Find discrete extrema
    fn discrete_extrema<F>(f: &F, xgrid: &[i64]) -> Vec<i64> 
    where 
        F: Fn(i64) -> f64,
    {
        if xgrid.len() < 3 {
            return Vec::new();
        }
        
        let fx: Vec<f64> = xgrid.iter().map(|&x| f(x)).collect();
        let mut extrema = Vec::new();
        
        // Check for extrema (local maxima or minima)
        for i in 1..fx.len() - 1 {
            let prev = fx[i - 1];
            let curr = fx[i];
            let next = fx[i + 1];
            
            // Local maximum
            if curr > prev && curr > next {
                extrema.push(xgrid[i]);
            }
            // Local minimum
            else if curr < prev && curr < next {
                extrema.push(xgrid[i]);
            }
        }
        
        extrema
    }
    
    /// Symmetrize Matsubara indices (remove zero if present and add negatives)
    fn symmetrize_matsubara_inplace(xs: &mut Vec<i64>) {
        // Remove zero if present
        xs.retain(|&x| x != 0);
        
        // Sort in ascending order
        xs.sort();
        
        // Create negative counterparts
        let negatives: Vec<i64> = xs.iter().rev().map(|&x| -x).collect();
        
        // Combine negatives with originals
        xs.splice(0..0, negatives);
    }
    
    /// Get T_nl coefficient (special function)
    /// 
    /// This implements the T_nl function which is related to spherical Bessel functions:
    /// T_nl(w) = 2 * i^l * j_l(|w|) * (w < 0 ? conj : identity)
    /// where j_l is the spherical Bessel function of the first kind.
    fn get_tnl(l: i32, w: f64) -> Complex64 {
        let abs_w = w.abs();
        
        // Compute spherical Bessel function j_l(abs_w)
        let sph_bessel = Self::spherical_bessel_j(l, abs_w);
        
        // Compute 2 * i^l
        let im_unit = Complex64::new(0.0, 1.0);
        let im_power = im_unit.powi(l);
        let result = 2.0 * im_power * sph_bessel;
        
        // Apply conjugation for negative w
        if w < 0.0 {
            result.conj()
        } else {
            result
        }
    }
    
    /// Spherical Bessel function of the first kind j_n(x)
    /// 
    /// This is a simplified implementation using series expansion and asymptotic forms.
    /// For production use, consider using a specialized library like `special-functions`.
    fn spherical_bessel_j(n: i32, x: f64) -> f64 {
        if x.abs() < 1e-10 {
            // Small argument expansion
            Self::spherical_bessel_j_small_x(n, x)
        } else if x > 50.0 {
            // Large argument asymptotic expansion
            Self::spherical_bessel_j_large_x(n, x)
        } else {
            // Medium argument - use series expansion
            Self::spherical_bessel_j_series(n, x)
        }
    }
    
    /// Spherical Bessel function for small arguments
    fn spherical_bessel_j_small_x(n: i32, x: f64) -> f64 {
        if n < 0 {
            return 0.0;
        }
        
        // Series expansion: j_n(x) ≈ x^n / (2n+1)!! * (1 - x²/(2n+3) + ...)
        let x_sq = x * x;
        let double_factorial = Self::double_factorial(2 * n + 1);
        
        if n == 0 {
            1.0 - x_sq / 6.0 + x_sq * x_sq / 120.0
        } else if n == 1 {
            x / 3.0 * (1.0 - x_sq / 10.0 + x_sq * x_sq / 280.0)
        } else {
            let x_power = x.powi(n);
            x_power / double_factorial * (1.0 - x_sq / (2 * n + 3) as f64)
        }
    }
    
    /// Spherical Bessel function for large arguments
    fn spherical_bessel_j_large_x(n: i32, x: f64) -> f64 {
        // Asymptotic expansion: j_n(x) ≈ sin(x - n*π/2) / x
        let phase = x - (n as f64) * PI / 2.0;
        phase.sin() / x
    }
    
    /// Spherical Bessel function using series expansion
    fn spherical_bessel_j_series(n: i32, x: f64) -> f64 {
        if n < 0 {
            return 0.0;
        }
        
        let x_sq = x * x;
        let mut sum = 0.0;
        let mut term = 1.0;
        
        // Series expansion up to reasonable precision
        for k in 0..20 {
            if k == 0 {
                term = x.powi(n) / Self::double_factorial(2 * n + 1);
            } else {
                term *= -x_sq / ((2 * k) * (2 * n + 2 * k + 1)) as f64;
            }
            
            sum += term;
            
            if term.abs() < 1e-15 * sum.abs() {
                break;
            }
        }
        
        sum
    }
    
    /// Double factorial: n!!
    fn double_factorial(n: i32) -> f64 {
        if n <= 0 {
            1.0
        } else if n % 2 == 0 {
            // Even: n!! = 2^(n/2) * (n/2)!
            let half_n = n / 2;
            2.0_f64.powi(half_n) * Self::factorial(half_n)
        } else {
            // Odd: n!! = n! / (2^((n-1)/2) * ((n-1)/2)!)
            let half_n_minus_1 = (n - 1) / 2;
            Self::factorial(n) / (2.0_f64.powi(half_n_minus_1) * Self::factorial(half_n_minus_1))
        }
    }
    
    /// Factorial: n!
    fn factorial(n: i32) -> f64 {
        if n <= 1 {
            1.0
        } else {
            let mut result = 1.0;
            for i in 2..=n {
                result *= i as f64;
            }
            result
        }
    }
}

/// Vector of PiecewiseLegendreFT polynomials
#[derive(Debug)]
pub struct PiecewiseLegendreFTVector<S: StatisticsType> {
    pub polyvec: Vec<PiecewiseLegendreFT<S>>,
    _phantom: std::marker::PhantomData<S>,
}

// Type aliases for convenience
pub type FermionicPiecewiseLegendreFTVector = PiecewiseLegendreFTVector<Fermionic>;
pub type BosonicPiecewiseLegendreFTVector = PiecewiseLegendreFTVector<Bosonic>;

impl<S: StatisticsType> PiecewiseLegendreFTVector<S> {
    /// Create an empty vector
    pub fn new() -> Self {
        Self {
            polyvec: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Create from a vector of PiecewiseLegendreFT
    pub fn from_vector(polyvec: Vec<PiecewiseLegendreFT<S>>) -> Self {
        Self {
            polyvec,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Create from PiecewiseLegendrePolyVector and statistics
    pub fn from_poly_vector(
        polys: &PiecewiseLegendrePolyVector,
        _stat: S,
        n_asymp: Option<f64>,
    ) -> Self {
        let mut polyvec = Vec::with_capacity(polys.size());
        
        for i in 0..polys.size() {
            let poly = polys.get(i).unwrap().clone();
            let ft_poly = PiecewiseLegendreFT::new(poly, _stat, n_asymp);
            polyvec.push(ft_poly);
        }
        
        Self {
            polyvec,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the size of the vector
    pub fn size(&self) -> usize {
        self.polyvec.len()
    }
    
    /// Get element by index (immutable)
    pub fn get(&self, index: usize) -> Option<&PiecewiseLegendreFT<S>> {
        self.polyvec.get(index)
    }
    
    /// Get element by index (mutable)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut PiecewiseLegendreFT<S>> {
        self.polyvec.get_mut(index)
    }
    
    /// Set element at index
    pub fn set(&mut self, index: usize, poly: PiecewiseLegendreFT<S>) -> Result<(), String> {
        if index >= self.polyvec.len() {
            return Err(format!("Index {} out of range", index));
        }
        self.polyvec[index] = poly;
        Ok(())
    }
    
    /// Create a similar empty vector
    pub fn similar(&self) -> Self {
        Self::new()
    }
    
    /// Get n_asymp from the first element (if any)
    pub fn n_asymp(&self) -> f64 {
        self.polyvec.first().map_or(f64::INFINITY, |p| p.n_asymp)
    }
    
    /// Evaluate all polynomials at a Matsubara frequency
    pub fn evaluate_at(&self, omega: &MatsubaraFreq<S>) -> Vec<Complex64> {
        self.polyvec.iter().map(|poly| poly.evaluate(omega)).collect()
    }
    
    /// Evaluate all polynomials at multiple Matsubara frequencies
    pub fn evaluate_at_many(&self, omegas: &[MatsubaraFreq<S>]) -> Vec<Vec<Complex64>> {
        omegas.iter().map(|omega| self.evaluate_at(omega)).collect()
    }
}

// Indexing operators
impl<S: StatisticsType> std::ops::Index<usize> for PiecewiseLegendreFTVector<S> {
    type Output = PiecewiseLegendreFT<S>;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.polyvec[index]
    }
}

impl<S: StatisticsType> std::ops::IndexMut<usize> for PiecewiseLegendreFTVector<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.polyvec[index]
    }
}

// Default implementations
impl<S: StatisticsType> Default for PiecewiseLegendreFTVector<S> {
    fn default() -> Self {
        Self::new()
    }
}

