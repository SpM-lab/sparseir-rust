//! SVE result container

use ndarray::Array1;
use crate::poly::PiecewiseLegendrePolyVector;

/// Result of Singular Value Expansion computation
#[derive(Debug, Clone)]
pub struct SVEResult {
    /// Left singular functions (u)
    pub u: PiecewiseLegendrePolyVector,
    /// Singular values in non-increasing order
    pub s: Array1<f64>,
    /// Right singular functions (v)
    pub v: PiecewiseLegendrePolyVector,
    /// Accuracy parameter used for computation
    pub epsilon: f64,
}

impl SVEResult {
    /// Create a new SVEResult
    pub fn new(
        u: PiecewiseLegendrePolyVector,
        s: Array1<f64>,
        v: PiecewiseLegendrePolyVector,
        epsilon: f64,
    ) -> Self {
        Self { u, s, v, epsilon }
    }

    /// Extract a subset of the SVE result based on epsilon and max_size
    /// 
    /// # Arguments
    /// 
    /// * `eps` - Relative threshold for singular values (default: self.epsilon)
    /// * `max_size` - Maximum number of singular values to keep
    /// 
    /// # Returns
    /// 
    /// Tuple of (u_subset, s_subset, v_subset)
    pub fn part(
        &self,
        eps: Option<f64>,
        max_size: Option<usize>,
    ) -> (PiecewiseLegendrePolyVector, Array1<f64>, PiecewiseLegendrePolyVector) {
        let eps = eps.unwrap_or(self.epsilon);
        let threshold = eps * self.s[0];

        let mut cut = 0;
        for &val in self.s.iter() {
            if val >= threshold {
                cut += 1;
            } else {
                break;
            }
        }

        if let Some(max) = max_size {
            cut = cut.min(max);
        }

        // Extract subsets
        let u_part = PiecewiseLegendrePolyVector::new(
            self.u.get_polys()[..cut].to_vec()
        );
        let s_part = self.s.slice(ndarray::s![..cut]).to_owned();
        let v_part = PiecewiseLegendrePolyVector::new(
            self.v.get_polys()[..cut].to_vec()
        );

        (u_part, s_part, v_part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::PiecewiseLegendrePoly;
    use ndarray::{Array2, array};

    fn create_dummy_poly() -> PiecewiseLegendrePoly {
        let data = Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap();
        let knots = vec![-1.0, 1.0];
        let delta_x = vec![2.0];
        PiecewiseLegendrePoly::new(data, knots, 0, Some(delta_x), 0)
    }

    #[test]
    fn test_part_with_threshold() {
        let u = PiecewiseLegendrePolyVector::new(vec![
            create_dummy_poly(),
            create_dummy_poly(),
            create_dummy_poly(),
        ]);
        let s = array![10.0, 5.0, 0.5];
        let v = u.clone();
        
        let result = SVEResult::new(u, s, v, 1e-8);
        
        // With threshold 0.1, should keep values >= 1.0
        let (_, s_part, _) = result.part(Some(0.1), None);
        assert_eq!(s_part.len(), 2);
        assert_eq!(s_part[0], 10.0);
        assert_eq!(s_part[1], 5.0);
    }

    #[test]
    fn test_part_with_max_size() {
        let u = PiecewiseLegendrePolyVector::new(vec![
            create_dummy_poly(),
            create_dummy_poly(),
            create_dummy_poly(),
        ]);
        let s = array![10.0, 5.0, 2.0];
        let v = u.clone();
        
        let result = SVEResult::new(u, s, v, 1e-8);
        
        // Max size limits output
        let (_, s_part, _) = result.part(Some(0.01), Some(2));
        assert_eq!(s_part.len(), 2);
    }

    #[test]
    fn test_part_default_epsilon() {
        let u = PiecewiseLegendrePolyVector::new(vec![
            create_dummy_poly(),
            create_dummy_poly(),
        ]);
        let s = array![10.0, 5.0];
        let v = u.clone();
        
        let epsilon = 0.6;  // threshold = 0.6 * 10.0 = 6.0
        let result = SVEResult::new(u, s, v, epsilon);
        
        let (_, s_part, _) = result.part(None, None);
        assert_eq!(s_part.len(), 1);  // Only 10.0 >= 6.0
    }
}

