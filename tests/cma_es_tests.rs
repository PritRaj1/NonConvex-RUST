mod common;

use nalgebra::{OMatrix, OVector, U20, U2, U1};
use common::fcns::{QuadraticObjective, QuadraticConstraints};

use non_convex_opt::algorithms::cma_es::cma_es::CMAES;
use non_convex_opt::utils::{
    config::CMAESConf,
    opt_prob::{OptProb, OptimizationAlgorithm},
};

#[test]
fn test_cmaes_initialization() {
    let conf = CMAESConf {
        population_size: 20,
        num_parents: 10,
        initial_sigma: 0.5,
    };

    let init_x = OMatrix::<f64, U1, U2>::from_element_generic(U1, U2, 0.5);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    
    let cmaes = CMAES::new(conf, init_x.clone(), opt_prob);
    
    let pop: OMatrix<f64, U20, U2> = cmaes.st.pop;
    let fit: OVector<f64, U20> = cmaes.st.fitness;
    let constr: OVector<bool, U20> = cmaes.st.constraints;

    // Check dims - these tests are redundant with statically-sized vectors
    assert_eq!(pop.nrows(), 20);
    assert_eq!(pop.ncols(), 2);
    assert_eq!(fit.len(), 20);
    assert_eq!(constr.len(), 20);
    
    // Check init
    assert_eq!(cmaes.st.best_x, init_x);
}

#[test]
fn test_cmaes() {
    let conf = CMAESConf {
        population_size: 20,
        num_parents: 10,
        initial_sigma: 0.3,
    };

    let init_x = OMatrix::<f64, U1, U2>::from_element_generic(U1, U2, 0.5);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    
    let mut cmaes:CMAES<f64, U20, U2> = CMAES::new(conf, init_x.clone(), opt_prob);
    
    for _ in 0..20 {
        cmaes.step();
        assert!(cmaes.st.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0),
            "Best solution violated constraints: {:?}", cmaes.st.best_x);
    }
}
