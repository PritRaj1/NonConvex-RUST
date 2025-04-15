use nalgebra::DVector;
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};

#[derive(Clone)]
pub struct KBF;

impl ObjectiveFunction<f64> for KBF {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let sum_cos4: f64 = x.iter().map(|&xi| xi.cos().powi(4)).sum();
        let prod_cos2: f64 = x.iter().map(|&xi| xi.cos().powi(2)).product();
        let sum_ix2: f64 = x.iter().enumerate().map(|(i, &xi)| (i as f64 + 1.0) * xi * xi).sum();
        (sum_cos4 - 2.0 * prod_cos2).abs() / sum_ix2.sqrt()
    }
}

#[derive(Clone)]
pub struct KBFConstraints;

impl BooleanConstraintFunction<f64> for KBFConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        let n = x.len();
        let product: f64 = x.iter().product();
        let sum: f64 = x.iter().sum();
        
        x.iter().all(|&xi| xi >= 0.0 && xi <= 10.0) &&
        product > 0.75 &&
        sum < (15.0 * n as f64) / 2.0
    }
} 

#[derive(Clone)]
pub struct RosenbrockFunction;

impl ObjectiveFunction<f64> for RosenbrockFunction {
    fn f(&self, x: &DVector<f64>) -> f64 {
        if x.len() < 2 {
            return f64::NEG_INFINITY;
        }
        let mut sum = 0.0;
        for i in 0..x.len()-1 {
            sum += 100.0 * (x[i+1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum  
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        
        // Catch empty vector
        if x.len() < 2 {
            return Some(DVector::zeros(x.len()));
        }

        let mut grad = DVector::zeros(x.len());
        grad[0] = -400.0 * x[0] * (x[1] - x[0].powi(2)) - 2.0 * (1.0 - x[0]);
        
        for i in 1..x.len()-1 {
            grad[i] = 200.0 * (x[i] - x[i-1].powi(2)) 
                     - 400.0 * x[i] * (x[i+1] - x[i].powi(2)) 
                     - 2.0 * (1.0 - x[i]);
        }
        
        let n = x.len() - 1;
        grad[n] = 200.0 * (x[n] - x[n-1].powi(2));
        Some(grad)
    }
}

#[derive(Debug, Clone)]
pub struct RosenbrockConstraints;

impl BooleanConstraintFunction<f64> for RosenbrockConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        x.iter().all(|&xi| xi >= -5.0 && xi <= 5.0)
    }
}