mod common;
use non_convex_opt::utils::config::TabuConf;
use non_convex_opt::utils::opt_prob::OptProb;
use non_convex_opt::tabu_search::tabu::TabuSearch;
use common::fcns::{RosenbrockObjective, RosenbrockConstraints};
use nalgebra::DVector;

#[test]
fn test_standard_tabu() {
    let conf = TabuConf {
        num_neighbors: 50,
        step_size: 0.1,
        perturbation_prob: 0.3,
        tabu_list_size: 20,
        tabu_threshold: 1e-6,
        tabu_type: "Standard".to_string(),
        min_tabu_size: 10,
        max_tabu_size: 30,
        increase_factor: 1.1,
        decrease_factor: 0.9,
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut tabu = TabuSearch::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = tabu.best_fitness;
    
    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.best_fitness > initial_fitness);
    assert!(tabu.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_reactive_tabu() {
    let conf = TabuConf {
        num_neighbors: 50,
        step_size: 0.1,
        perturbation_prob: 0.3,
        tabu_list_size: 20,
        tabu_threshold: 1e-6,
        tabu_type: "Reactive".to_string(),
        min_tabu_size: 10,
        max_tabu_size: 30,
        increase_factor: 1.1,
        decrease_factor: 0.9,
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut tabu = TabuSearch::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = tabu.best_fitness;
    
    for _ in 0..10 {
        tabu.step();
    }

    assert!(tabu.best_fitness > initial_fitness);
    assert!(tabu.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
} 