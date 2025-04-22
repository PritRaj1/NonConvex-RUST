mod common;
use non_convex_opt::cma_es::cma_es::CMAES;
use non_convex_opt::utils::config::CMAESConf;
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{QuadraticObjective, QuadraticConstraints};
use nalgebra::DVector;

#[test]
fn test_cmaes_initialization() {
    let conf = CMAESConf {
        population_size: 20,
        num_parents: 10,
        initial_sigma: 0.5,
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let cmaes = CMAES::new(conf, init_x.clone(), opt_prob);
    
    // Check dims
    assert_eq!(cmaes.population.nrows(), 20);
    assert_eq!(cmaes.population.ncols(), 2);
    assert_eq!(cmaes.fitness.len(), 20);
    assert_eq!(cmaes.constraints.len(), 20);
    
    // Check init
    assert_eq!(cmaes.x, init_x);
}

#[test]
fn test_cmaes() {
    let conf = CMAESConf {
        population_size: 20,
        num_parents: 10,
        initial_sigma: 0.3,
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut cmaes = CMAES::new(conf, init_x, opt_prob);
    
    for _ in 0..20 {
        cmaes.step();
        assert!(cmaes.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0),
            "Best solution violated constraints: {:?}", cmaes.best_x);
    }
}
