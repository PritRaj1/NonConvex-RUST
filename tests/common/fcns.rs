use nalgebra::DVector;
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};

#[derive(Debug, Clone)]
pub struct RosenbrockObjective {
    pub a: f64,
    pub b: f64,
}

impl ObjectiveFunction<f64> for RosenbrockObjective {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..n-1 {
            sum += self.b * (x[i+1] - x[i].powi(2)).powi(2) + 
                   (self.a - x[i]).powi(2);
        }
        sum
    }
}

#[derive(Debug, Clone)]
pub struct RosenbrockConstraints {}

impl BooleanConstraintFunction<f64> for RosenbrockConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        x.iter().all(|&xi| xi >= 0.0 && xi <= 1.0)
    }
}

#[derive(Debug, Clone)]
pub struct QuadraticObjective {
    pub a: f64,
    pub b: f64,
}

impl ObjectiveFunction<f64> for QuadraticObjective {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..n {
            sum -= self.b * x[i].powi(2) - self.a * x[i];
        }
        sum
    }

    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        for i in 0..n {
            grad[i] = -2.0 * self.b * x[i] + self.a;
        }
        Some(grad)
    }

    fn x_lower_bound(&self) -> Option<DVector<f64>> {
        Some(DVector::from_element(2, 0.0))
    }

    fn x_upper_bound(&self) -> Option<DVector<f64>> {
        Some(DVector::from_element(2, 1.0))
    }
}

#[derive(Debug, Clone)]
pub struct QuadraticConstraints {}

impl BooleanConstraintFunction<f64> for QuadraticConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        x.iter().all(|&xi| xi >= 0.0 && xi <= 1.0)
    }
}   