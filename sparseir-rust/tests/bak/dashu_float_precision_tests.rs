use dashu_float::{DBig, Context};
use dashu_float::round::mode::HalfAway;
use dashu_base::Approximation::{Exact, Inexact};
use dashu_base::Approximation;
use dashu_base::Abs;
use std::str::FromStr;
use libm;

mod dashu_float_precision_tests {
    use super::*;

    // Test configuration
    const HIGH_PRECISION_DIGITS: usize = 200;

    /// Helper function to create high-precision context
    fn high_precision_ctx() -> Context<HalfAway> {
        Context::<HalfAway>::new(HIGH_PRECISION_DIGITS)
    }

    /// Helper function to convert f64 to high-precision DBig
    fn f64_to_dbig(val: f64, precision: usize) -> DBig {
        let val_str = format!("{:.34e}", val);
        DBig::from_str(&val_str)
            .unwrap()
            .with_precision(precision)
            .unwrap()
    }

    /// Helper function to extract f64 from Approximation
    fn extract_f64(approx: Approximation<f64, dashu_float::round::Rounding>) -> f64 {
        match approx {
            Exact(val) => val,
            Inexact(val, _) => val,
        }
    }

    /// Helper function to convert TwoFloat to DBig using hi and lo components with maximum precision
    fn twofloat_to_dbig(val: twofloat::TwoFloat, precision: usize) -> DBig {
        // Access TwoFloat's internal hi and lo components
        let hi = val.hi();
        let lo = val.lo();
        
        // Use higher precision string formatting to preserve more digits
        let hi_str = format!("{:.34e}", hi);
        let lo_str = format!("{:.34e}", lo);
        
        // Parse with higher precision context
        let ctx_high = Context::<HalfAway>::new(precision); 
        
        let hi_dbig = DBig::from_str(&hi_str)
            .unwrap()
            .with_precision(precision)
            .unwrap();
        let lo_dbig = DBig::from_str(&lo_str)
            .unwrap()
            .with_precision(precision)
            .unwrap();
        
        // Add hi + lo with high precision
        let sum = hi_dbig + lo_dbig;
        
        // Return with the target precision
        sum.with_precision(precision).unwrap()
    }

    #[test]
    fn test_twofloat_vs_dbig() {
        use twofloat::TwoFloat;
        use sparseir_rust::CustomNumeric;
        //use twofloat::{E};
        
        println!("TwoFloat vs DBig exp(1) precision test:");
        println!("========================================");

        let a = TwoFloat::from(1.0);
        let b = a.exp();
        let diff = (b - twofloat::consts::E)/twofloat::consts::E;
        println!("diff: {}", diff);
        println!("diff * 1e+18: {}", diff * 1e+18);
        assert!((b - twofloat::consts::E)/twofloat::consts::E < 1e-30);

        let precision: usize = 200;
        
        let ctx_100 = Context::<HalfAway>::new(precision);
        let val = 1.0;
        let dbig_val = f64_to_dbig(val, precision);
        //let dbig_100 = dbig_val.with_precision(precision).unwrap();
        let dbig_result = match ctx_100.exp(dbig_val.repr()) {
            Exact(result) => result,
            Inexact(result, _) => result,
        };
        
        println!("DBig reference (200 digits):");
        println!("{}", dbig_result.to_string());
        println!();
        
        // TwoFloat calculation
        let twofloat_val = TwoFloat::from_f64(val);
        let twofloat_result = twofloat_val.exp();
        //let twofloat_result = twofloat::E;
        println!("hi: {:40e}", twofloat_result.hi());
        println!("lo: {:40e}", twofloat_result.lo());

        println!("TwoFloat result:");
        println!("{}", twofloat_result.to_f64());
        println!();
        
        // Convert TwoFloat to DBig for comparison
        let twofloat_as_dbig = twofloat_to_dbig(twofloat_result, 100);
        
        // Calculate difference
        let diff = (&dbig_result - &twofloat_as_dbig).abs();
        println!("{:.34e}", extract_f64(diff.to_f64()));

        let v = TwoFloat::from_f64(1.0);
        let mv = - v;
        let diff = v.exp() * mv.exp() - TwoFloat::from_f64(1.0);

        println!("diff {} {}", diff.hi(), diff.lo());
        println!("diff*1e+10 {} {}", diff.hi() * 1e+10, diff.lo() * 1e+10);

        panic!("fail")
        //let rel_error = if dbig_result == DBig::ZERO {
            //extract_f64(diff.to_f64())
        //} else {
            //let diff_f64 = extract_f64(diff.to_f64());
            //let dbig_f64 = extract_f64(dbig_result.to_f64());
            //diff_f64 / dbig_f64.abs()
        //};
        //
        //println!("Precision comparison:");
        //println!("  DBig (100 digits):    {}", dbig_result.to_string());
        //println!("  TwoFloat:             {}", twofloat_result.to_f64());
        //println!("  TwoFloat as DBig:     {}", twofloat_as_dbig_100.to_string());
        //println!("  Absolute difference:  {:.2e}", extract_f64(diff.to_f64()));
        //println!("  Relative error:       {:.2e}", rel_error);
        //
        //// Determine effective precision of TwoFloat
        //let effective_precision = -rel_error.log10();
        //println!("  TwoFloat effective precision: ~{:.1} digits", effective_precision);
    }
}