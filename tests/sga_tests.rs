mod common;
use non_convex_opt::sg_ascent::sga::SGAscent;
use non_convex_opt::utils::config::SGAConf;
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{QuadraticObjective, QuadraticConstraints};
use nalgebra::DVector;

#[test]
fn test_sga() {
    let conf = SGAConf {
        learning_rate: 0.01,
        momentum: 0.9,
    };

    let init_x = DVector::from_vec(vec![1.0, 1.0]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut sga = SGAscent::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = sga.best_fitness;
    
    for _ in 0..10 {
        sga.step();
    }

    assert!(sga.best_fitness > initial_fitness);
    assert!(sga.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
} 