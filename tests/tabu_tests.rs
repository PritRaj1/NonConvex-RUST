mod common;
use non_convex_opt::utils::config::TabuConf;
use non_convex_opt::utils::opt_prob::OptProb;
use non_convex_opt::tabu_search::tabu::TabuSearch;
use common::fcns::{RosenbrockObjective, RosenbrockConstraints};
use nalgebra::DVector;

#[test]
fn test_tabu_search() {
    let conf = TabuConf {
        tabu_list_size: 20,
        num_neighbors: 50,
        step_size: 0.1,
        perturbation_prob: 0.3,
        tabu_threshold: 1e-6,
    };

    let init_x = DVector::from_vec(vec![0.0, 0.0]);
    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut tabu = TabuSearch::new(conf, init_x.clone(), opt_prob);
    
    let initial_fitness = tabu.best_fitness;
    
    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.best_fitness > initial_fitness);
    assert!(tabu.best_x.iter().all(|&x| x.abs() <= 2.0));
} 