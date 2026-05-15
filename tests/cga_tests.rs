mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{OMatrix, U10, U2};

use non_convex_opt::utils::config::OptConf;
use non_convex_opt::utils::{
    config::{AlgConf, Config},
    opt_prob::{ObjectiveFunction, OptProb, OptimizationAlgorithm},
};

use non_convex_opt::algorithms::continuous_genetic::CGA;

#[test]
fn test_adaptive_parameters() {
    let conf = Config::new(include_str!("jsons/cga.json")).unwrap();
    let cga_conf = match conf.alg_conf {
        AlgConf::CGA(c) => c,
        _ => panic!("Expected CGAConf"),
    };

    let mut init_pop = OMatrix::zeros_generic(U10, U2);
    for i in 0..10 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0;
        }
    }

    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mut cga = CGA::new(cga_conf, init_pop, opt_prob, &OptConf::default(), 42);
    for _ in 0..10 {
        cga.step();
    }

    let (mut_rate, cross_prob) = cga.get_current_parameters();
    assert!(mut_rate.is_finite() && (0.001..=0.5).contains(&mut_rate));
    assert!(cross_prob.is_finite() && (0.1..=0.95).contains(&cross_prob));
}

#[test]
fn test_cga() {
    let conf = Config::new(include_str!("jsons/cga.json")).unwrap();
    let cga_conf = match conf.alg_conf {
        AlgConf::CGA(c) => c,
        _ => panic!("Expected CGAConf"),
    };

    let mut init_pop = OMatrix::zeros_generic(U10, U2);
    for i in 0..10 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0;
        }
    }

    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let initial = init_pop.row(0).clone_owned();
    let initial_fitness = RosenbrockObjective { a: 1.0, b: 1.0 }
        .f(&initial.transpose());

    let mut cga = CGA::new(cga_conf, init_pop, opt_prob, &OptConf::default(), 42);
    for _ in 0..30 {
        cga.step();
    }
    assert!(cga.st.best_f.is_finite());
    assert!(cga.st.best_f >= initial_fitness);
}
