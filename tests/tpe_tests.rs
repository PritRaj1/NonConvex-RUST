mod common;

use common::fcns::{RosenbrockConstraints, RosenbrockObjective};
use nalgebra::{SMatrix, U2, U5};

use non_convex_opt::algorithms::tpe::{TPEConf, TPE};
use non_convex_opt::utils::config::OptConf;
use non_convex_opt::utils::opt_prob::{OptProb, OptimizationAlgorithm};

#[test]
fn test_tpe_improves() {
    let conf = TPEConf {
        n_initial_random: 10,
        n_candidates: 50,
        gamma: 0.25,
        max_history: 200,
    };

    let init_pop = SMatrix::<f64, 5, 2>::from_rows(&[
        [0.9, 0.9].into(),
        [0.8, 0.8].into(),
        [0.7, 0.7].into(),
        [0.6, 0.6].into(),
        [0.5, 0.5].into(),
    ]);

    let obj_f = RosenbrockObjective { a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let mut tpe: TPE<f64, U5, U2> = TPE::new(
        conf,
        init_pop,
        opt_prob,
        &OptConf {
            stagnation_window: 50,
            ..OptConf::default()
        },
        42,
    );
    let initial_fitness = tpe.st.best_f;
    for _ in 0..30 {
        tpe.step();
    }
    assert!(tpe.st.best_f >= initial_fitness);
    assert!(tpe.st.iter > 1);
}
