mod common;
use non_convex_opt::grasp::grasp::GRASP;
use non_convex_opt::utils::config::GRASPConf;
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{RosenbrockObjective, RosenbrockConstraints};
use nalgebra::DVector;

#[test]
fn test_grasp_unconstrained() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
    };

    let init_x = DVector::from_vec(vec![0.0, 0.0]);
    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let opt_prob = OptProb::new(obj_f, None::<RosenbrockConstraints>);
    
    let mut grasp = GRASP::new(conf, init_x.clone(), opt_prob);
    
    let initial_fitness = grasp.best_fitness;
    
    for _ in 0..10 {
        grasp.step();
    }

    assert!(grasp.best_fitness > initial_fitness);
}

#[test]
fn test_grasp_constrained() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut grasp = GRASP::new(conf, init_x.clone(), opt_prob);
    
    let initial_fitness = grasp.best_fitness;
    
    for _ in 0..10 {
        grasp.step();
    }

    assert!(grasp.best_fitness > initial_fitness);
    assert!(grasp.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_grasp_construction_and_local_search() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let grasp = GRASP::new(conf, init_x.clone(), opt_prob);
    
    let solution = grasp.construct_solution();
    let improved = grasp.local_search(&solution);
    
    assert!(grasp.opt_prob.is_feasible(&solution));
    assert!(grasp.opt_prob.is_feasible(&improved));
}

#[test]
fn test_grasp_bounds() {
    let conf = GRASPConf {
        num_candidates: 50,
        alpha: 0.3,
        num_neighbors: 20,
        step_size: 0.1,
        perturbation_prob: 0.3,
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut grasp = GRASP::new(conf, init_x.clone(), opt_prob);
    
    for _ in 0..10 {
        grasp.step();
        assert!(grasp.x.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
} 