mod common;

use nalgebra::DVector;
use common::fcns::{QuadraticObjective, QuadraticConstraints};
use non_convex_opt::algorithms::sg_ascent::sga::SGAscent;
use non_convex_opt::utils::{
    config::SGAConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_sga() {
    let conf = SGAConf {
        learning_rate: 0.01,
        momentum: 0.9,
    };

    let init_x = DMatrix::from_columns(vec![1.0, 1.0]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    
    let mut sga = SGAscent::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = sga.st.best_fitness;
    
    for _ in 0..10 {
        sga.step();
    }

    assert!(sga.st.best_f > initial_fitness);
    assert!(sga.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
} 