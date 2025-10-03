//! Piecewise Legendre polynomial implementations for SparseIR
//!
//! This module provides high-performance piecewise Legendre polynomial
//! functionality with integrated data storage for optimal memory efficiency.

use std::sync::Arc;
use twofloat::TwoFloat;

/// Metadata for a single polynomial within the unified data structure
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialInfo {
    /// Start index in the unified coefficient array
    pub coefficient_start: usize,
    /// Number of coefficients for this polynomial
    pub coefficient_count: usize,
    /// Start index in the unified interval array
    pub interval_start: usize,
    /// Number of intervals for this polynomial
    pub interval_count: usize,
    /// Degree of this polynomial
    pub degree: usize,
}

/// Unified data structure containing all polynomial information
#[derive(Debug)]
pub struct PolynomialData {
    /// All coefficients from all polynomials stored contiguously
    pub all_coefficients: Vec<f64>,
    /// All intervals from all polynomials stored contiguously
    pub all_intervals: Vec<(f64, f64)>,
    /// Metadata for each polynomial
    pub polynomial_info: Vec<PolynomialInfo>,
}

impl PolynomialData {
    /// Create a new PolynomialData from individual polynomial data
    pub fn from_individual_polynomials(
        coefficients: Vec<Vec<f64>>,
        intervals: Vec<Vec<(f64, f64)>>,
        degrees: Vec<usize>,
    ) -> Self {
        let mut all_coefficients = Vec::new();
        let mut all_intervals = Vec::new();
        let mut polynomial_info = Vec::new();
        
        for i in 0..coefficients.len() {
            let coeff_start = all_coefficients.len();
            let coeff_count = coefficients[i].len();
            let interval_start = all_intervals.len();
            let interval_count = intervals[i].len();
            
            all_coefficients.extend_from_slice(&coefficients[i]);
            all_intervals.extend_from_slice(&intervals[i]);
            
            polynomial_info.push(PolynomialInfo {
                coefficient_start: coeff_start,
                coefficient_count: coeff_count,
                interval_start: interval_start,
                interval_count: interval_count,
                degree: degrees[i],
            });
        }
        
        Self {
            all_coefficients,
            all_intervals,
            polynomial_info,
        }
    }
    
    /// Get coefficient slice for a specific polynomial
    pub fn get_coefficients(&self, index: usize) -> Option<&[f64]> {
        let info = self.polynomial_info.get(index)?;
        let start = info.coefficient_start;
        let end = start + info.coefficient_count;
        Some(&self.all_coefficients[start..end])
    }
    
    /// Get interval slice for a specific polynomial
    pub fn get_intervals(&self, index: usize) -> Option<&[(f64, f64)]> {
        let info = self.polynomial_info.get(index)?;
        let start = info.interval_start;
        let end = start + info.interval_count;
        Some(&self.all_intervals[start..end])
    }
    
    /// Get polynomial info for a specific polynomial
    pub fn get_info(&self, index: usize) -> Option<&PolynomialInfo> {
        self.polynomial_info.get(index)
    }
}

/// A single piecewise Legendre polynomial
#[derive(Debug, Clone)]
pub struct PiecewiseLegendrePoly {
    /// Shared reference to the unified data
    data: Arc<PolynomialData>,
    /// Index of this polynomial within the unified data
    index: usize,
}

impl PiecewiseLegendrePoly {
    /// Create a new polynomial from individual data
    pub fn new(
        coefficients: Vec<f64>,
        intervals: Vec<(f64, f64)>,
        degree: usize,
    ) -> Self {
        let polynomial_data = PolynomialData::from_individual_polynomials(
            vec![coefficients],
            vec![intervals],
            vec![degree],
        );
        
        Self {
            data: Arc::new(polynomial_data),
            index: 0,
        }
    }
    
    /// Create a polynomial from shared data
    pub(crate) fn from_shared_data(data: Arc<PolynomialData>, index: usize) -> Self {
        Self { data, index }
    }
    
    /// Get the coefficients of this polynomial
    pub fn coefficients(&self) -> &[f64] {
        self.data.get_coefficients(self.index).unwrap()
    }
    
    /// Get the intervals of this polynomial
    pub fn intervals(&self) -> &[(f64, f64)] {
        self.data.get_intervals(self.index).unwrap()
    }
    
    /// Get the degree of this polynomial
    pub fn degree(&self) -> usize {
        self.data.get_info(self.index).unwrap().degree
    }
    
    /// Evaluate the polynomial at a given point with high precision
    pub fn evaluate(&self, x: TwoFloat) -> TwoFloat {
        let coefficients = self.coefficients();
        let intervals = self.intervals();
        
        // Find the interval containing x
        for (interval_idx, &(interval_start, interval_end)) in intervals.iter().enumerate() {
            let start = TwoFloat::from(interval_start);
            let end = TwoFloat::from(interval_end);
            
            if x >= start && x < end {
                return self.evaluate_in_interval(x, interval_idx, coefficients, intervals);
            }
        }
        
        // If x is exactly at the last interval's end point
        if let Some(&(_, last_end)) = intervals.last() {
            if x == TwoFloat::from(last_end) {
                return self.evaluate_in_interval(x, intervals.len() - 1, coefficients, intervals);
            }
        }
        
        // Outside domain - return zero
        TwoFloat::from(0.0)
    }
    
    /// Evaluate polynomial in a specific interval
    fn evaluate_in_interval(
        &self,
        x: TwoFloat,
        interval_idx: usize,
        coefficients: &[f64],
        intervals: &[(f64, f64)],
    ) -> TwoFloat {
        let (_interval_start, _interval_end) = intervals[interval_idx];
        
        // For piecewise polynomials, we evaluate directly in the original domain
        // without transforming to [-1, 1] unless specifically needed for Legendre polynomials
        let mut result = TwoFloat::from(0.0);
        let mut power = TwoFloat::from(1.0);
        
        for &coeff in coefficients {
            result += TwoFloat::from(coeff) * power;
            power *= x;
        }
        
        result
    }
    
    /// Evaluate the polynomial at a given point with f64 precision
    pub fn evaluate_f64(&self, x: f64) -> f64 {
        Into::<f64>::into(self.evaluate(TwoFloat::from(x)))
    }
    
    /// Get the domain range of this polynomial
    pub fn domain(&self) -> (f64, f64) {
        let intervals = self.intervals();
        if intervals.is_empty() {
            return (0.0, 0.0);
        }
        
        let first_start = intervals[0].0;
        let last_end = intervals.last().unwrap().1;
        (first_start, last_end)
    }
    
    /// Get the shared data (for advanced usage)
    pub fn shared_data(&self) -> &Arc<PolynomialData> {
        &self.data
    }
}

/// A vector of piecewise Legendre polynomials with integrated data storage
#[derive(Debug, Clone)]
pub struct PiecewiseLegendrePolyVector {
    /// Shared data containing all polynomial information
    shared_data: Arc<PolynomialData>,
    /// Array of polynomial references for easy access
    polynomials: Vec<PiecewiseLegendrePoly>,
}

impl PiecewiseLegendrePolyVector {
    /// Create a new vector from individual polynomials
    pub fn new(polynomials: Vec<PiecewiseLegendrePoly>) -> Self {
        if polynomials.is_empty() {
            return Self {
                shared_data: Arc::new(PolynomialData {
                    all_coefficients: Vec::new(),
                    all_intervals: Vec::new(),
                    polynomial_info: Vec::new(),
                }),
                polynomials: Vec::new(),
            };
        }
        
        // Extract data from individual polynomials
        let mut coefficients = Vec::new();
        let mut intervals = Vec::new();
        let mut degrees = Vec::new();
        
        for poly in &polynomials {
            coefficients.push(poly.coefficients().to_vec());
            intervals.push(poly.intervals().to_vec());
            degrees.push(poly.degree());
        }
        
        // Create unified data
        let shared_data = Arc::new(PolynomialData::from_individual_polynomials(
            coefficients,
            intervals,
            degrees,
        ));
        
        // Create new polynomial references pointing to shared data
        let polynomials: Vec<_> = (0..shared_data.polynomial_info.len())
            .map(|index| PiecewiseLegendrePoly::from_shared_data(Arc::clone(&shared_data), index))
            .collect();
        
        Self {
            shared_data,
            polynomials,
        }
    }
    
    /// Get the number of polynomials in this vector
    pub fn len(&self) -> usize {
        self.polynomials.len()
    }
    
    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.polynomials.is_empty()
    }
    
    /// Get a reference to a polynomial by index
    pub fn get(&self, index: usize) -> Option<&PiecewiseLegendrePoly> {
        self.polynomials.get(index)
    }
    
    /// Extract (clone) a polynomial by index
    pub fn extract(&self, index: usize) -> Option<PiecewiseLegendrePoly> {
        self.polynomials.get(index).cloned()
    }
    
    /// Get an iterator over all polynomials
    pub fn iter(&self) -> impl Iterator<Item = &PiecewiseLegendrePoly> {
        self.polynomials.iter()
    }
    
    /// Get the shared data (for advanced usage)
    pub fn shared_data(&self) -> &Arc<PolynomialData> {
        &self.shared_data
    }
}

impl std::ops::Index<usize> for PiecewiseLegendrePolyVector {
    type Output = PiecewiseLegendrePoly;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.polynomials[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_polynomial_data_creation() {
        let coefficients = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let intervals = vec![vec![(0.0, 1.0)], vec![(1.0, 2.0), (2.0, 3.0)]];
        let degrees = vec![1, 2];
        
        let data = PolynomialData::from_individual_polynomials(coefficients, intervals, degrees);
        
        assert_eq!(data.all_coefficients, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(data.all_intervals, vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]);
        assert_eq!(data.polynomial_info.len(), 2);
        
        // Check first polynomial info
        let info1 = data.get_info(0).unwrap();
        assert_eq!(info1.coefficient_start, 0);
        assert_eq!(info1.coefficient_count, 2);
        assert_eq!(info1.interval_start, 0);
        assert_eq!(info1.interval_count, 1);
        assert_eq!(info1.degree, 1);
        
        // Check second polynomial info
        let info2 = data.get_info(1).unwrap();
        assert_eq!(info2.coefficient_start, 2);
        assert_eq!(info2.coefficient_count, 3);
        assert_eq!(info2.interval_start, 1);
        assert_eq!(info2.interval_count, 2);
        assert_eq!(info2.degree, 2);
    }
    
    #[test]
    fn test_individual_polynomial() {
        let poly = PiecewiseLegendrePoly::new(
            vec![1.0, 2.0, 3.0],
            vec![(0.0, 1.0), (1.0, 2.0)],
            2,
        );
        
        assert_eq!(poly.coefficients(), &[1.0, 2.0, 3.0]);
        assert_eq!(poly.intervals(), &[(0.0, 1.0), (1.0, 2.0)]);
        assert_eq!(poly.degree(), 2);
        assert_eq!(poly.domain(), (0.0, 2.0));
    }
    
    #[test]
    fn test_polynomial_vector() {
        let poly1 = PiecewiseLegendrePoly::new(vec![1.0, 2.0], vec![(0.0, 1.0)], 1);
        let poly2 = PiecewiseLegendrePoly::new(vec![3.0, 4.0, 5.0], vec![(1.0, 2.0)], 2);
        
        let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);
        
        assert_eq!(vector.len(), 2);
        assert!(!vector.is_empty());
        
        // Test indexing
        assert_eq!(vector[0].coefficients(), &[1.0, 2.0]);
        assert_eq!(vector[1].coefficients(), &[3.0, 4.0, 5.0]);
        
        // Test extraction
        let extracted = vector.extract(0).unwrap();
        assert_eq!(extracted.coefficients(), &[1.0, 2.0]);
        
        // Test iteration
        let mut iter = vector.iter();
        assert_eq!(iter.next().unwrap().degree(), 1);
        assert_eq!(iter.next().unwrap().degree(), 2);
        assert!(iter.next().is_none());
    }
    
    #[test]
    fn test_polynomial_evaluation() {
        // Test constant polynomial: P(x) = 5.0
        let poly = PiecewiseLegendrePoly::new(vec![5.0], vec![(0.0, 1.0)], 0);
        
        assert_eq!(poly.evaluate_f64(0.0), 5.0);
        assert_eq!(poly.evaluate_f64(0.5), 5.0);
        assert_eq!(poly.evaluate_f64(1.0), 5.0);
        
        // Test linear polynomial: P(x) = 1.0 + 2.0*x
        let poly = PiecewiseLegendrePoly::new(vec![1.0, 2.0], vec![(0.0, 1.0)], 1);
        
        assert_eq!(poly.evaluate_f64(0.0), 1.0);
        assert_eq!(poly.evaluate_f64(0.5), 2.0);
        assert_eq!(poly.evaluate_f64(1.0), 3.0);
    }
    
    #[test]
    fn test_empty_vector() {
        let vector = PiecewiseLegendrePolyVector::new(vec![]);
        assert_eq!(vector.len(), 0);
        assert!(vector.is_empty());
        assert!(vector.get(0).is_none());
        assert!(vector.extract(0).is_none());
    }
    
    #[test]
    fn test_shared_data_efficiency() {
        let poly1 = PiecewiseLegendrePoly::new(vec![1.0, 2.0], vec![(0.0, 1.0)], 1);
        let poly2 = PiecewiseLegendrePoly::new(vec![3.0, 4.0], vec![(1.0, 2.0)], 1);
        
        let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);
        
        // Extract both polynomials
        let extracted1 = vector.extract(0).unwrap();
        let extracted2 = vector.extract(1).unwrap();
        
        // They should share the same underlying data
        assert_eq!(extracted1.coefficients(), &[1.0, 2.0]);
        assert_eq!(extracted2.coefficients(), &[3.0, 4.0]);
        
        // The Arc should be shared (same reference count)
        let shared_data1 = &extracted1.data;
        let shared_data2 = &extracted2.data;
        
        // Both should point to the same Arc
        assert!(Arc::ptr_eq(shared_data1, shared_data2));
    }
}
