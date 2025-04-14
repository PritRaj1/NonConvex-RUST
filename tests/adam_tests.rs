mod common;
use non_convex_opt::adam::adam::Adam;
use non_convex_opt::utils::config::AdamConf;
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{QuadraticObjective, QuadraticConstraints};
use nalgebra::DVector;

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
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut adam = Adam::new(conf, init_x.clone(), opt_prob);
    
    let initial_fitness = adam.best_fitness;
    
    for _ in 0..10 {
        adam.step();
    }

    assert!(adam.best_fitness > initial_fitness);
} 