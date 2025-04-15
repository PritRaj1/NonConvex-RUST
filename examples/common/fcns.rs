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

#[derive(Debug, Clone)]
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
pub struct MultiModalFunction;

impl ObjectiveFunction<f64> for MultiModalFunction {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let gaussian1 = -0.5 * ((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2));
        let gaussian2 = -0.3 * ((x[0] - 7.0).powi(2) + (x[1] - 7.0).powi(2));
        let gaussian3 = -0.2 * ((x[0] - 7.0).powi(2) + (x[1] - 3.0).powi(2));
        
        10.0 * (gaussian1.exp() + gaussian2.exp() + gaussian3.exp())
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut grad = DVector::zeros(2);
        
        let exp1 = (-0.5 * ((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2))).exp();
        grad[0] += 10.0 * exp1 * (-(x[0] - 3.0));
        grad[1] += 10.0 * exp1 * (-(x[1] - 3.0));
        
        let exp2 = (-0.3 * ((x[0] - 7.0).powi(2) + (x[1] - 7.0).powi(2))).exp();
        grad[0] += 6.0 * exp2 * (-(x[0] - 7.0));
        grad[1] += 6.0 * exp2 * (-(x[1] - 7.0));
        
        let exp3 = (-0.2 * ((x[0] - 7.0).powi(2) + (x[1] - 3.0).powi(2))).exp();
        grad[0] += 4.0 * exp3 * (-(x[0] - 7.0));
        grad[1] += 4.0 * exp3 * (-(x[1] - 3.0));
        
        Some(grad)
    }
}

#[derive(Debug, Clone)]
pub struct BoxConstraints;

impl BooleanConstraintFunction<f64> for BoxConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        x.iter().all(|&xi| xi >= 0.0 && xi <= 10.0)
    }
}