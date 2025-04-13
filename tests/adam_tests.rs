use non_convex_opt::adam::adam::Adam;
use non_convex_opt::utils::config::AdamConf;
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction, OptProb};
use nalgebra::DVector;

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
pub struct QuadraticConstraints {
    pub a: f64,
    pub b: f64,
}

impl BooleanConstraintFunction<f64> for QuadraticConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        // Check if all components are within bounds
        x.iter().all(|&xi| xi >= 0.0 && xi <= 1.0)
    }
}   

#[test]
fn test_adam() {
    let conf = AdamConf {
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };

    let init_x = DVector::from_vec(vec![1.0, 1.0]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints { a: 0.0, b: 1.0 };
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut adam = Adam::new(conf, init_x.clone(), opt_prob);
    
    let initial_fitness = adam.best_fitness;
    
    for _ in 0..10 {
        adam.step();
    }

    assert!(adam.best_fitness > initial_fitness);
} 