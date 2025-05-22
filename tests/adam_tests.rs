mod common;

use nalgebra::DMatrix;
use non_convex_opt::algorithms::adam::adam::Adam;
use common::fcns::{QuadraticObjective, QuadraticConstraints};
use non_convex_opt::utils::{
    config::AdamConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_adam() {
    let conf = AdamConf {
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };

    let init_x = DMatrix::from_row_slice(1, 2, &[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    
    let mut adam = Adam::new(conf, init_x.clone(), opt_prob);
    
    let initial_fitness = adam.st.best_f;
    
    for _ in 0..10 {
        adam.step();
    }

    assert!(adam.st.best_f > initial_fitness);
} 