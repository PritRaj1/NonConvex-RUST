mod common;
use non_convex_opt::differential_evolution::de::DE;
use non_convex_opt::utils::config::{DEConf, DEStrategy};
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{RosenbrockObjective, RosenbrockConstraints};
use nalgebra::DMatrix;

#[test]
fn test_de_basic() {
    let conf = DEConf {
        population_size: 50,
        f: 0.8,
        cr: 0.9,
        strategy: DEStrategy::Rand1Bin,
    };

    let mut init_pop = DMatrix::zeros(50, 2);
    for i in 0..50 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0; // Random values in [-2, 2]
        }
    }

    let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut de = DE::new(conf, init_pop.clone(), opt_prob);
    let initial_fitness = de.best_fitness;
    
    for _ in 0..10 {
        de.step();
    }

    assert!(de.best_fitness > initial_fitness);
}

#[test]
fn test_de_strategies() {
    let strategies = vec![
        DEStrategy::Rand1Bin,
        DEStrategy::Best1Bin,
        DEStrategy::RandToBest1Bin,
        DEStrategy::Best2Bin,
        DEStrategy::Rand2Bin,
    ];

    for strategy in strategies {
        let conf = DEConf {
            population_size: 50,
            f: 0.8,
            cr: 0.9,
            strategy: strategy.clone(),
        };

        let mut init_pop = DMatrix::zeros(50, 2);
        for i in 0..50 {
            for j in 0..2 {
                init_pop[(i, j)] = rand::random::<f64>() * 4.0 - 2.0;
            }
        }

        let obj_f = RosenbrockObjective{ a: 1.0, b: 1.0};
        let constraints = RosenbrockConstraints{};
        let opt_prob = OptProb::new(obj_f, Some(constraints));
        
        let mut de = DE::new(conf, init_pop.clone(), opt_prob);
        let initial_fitness = de.best_fitness;
        
        for _ in 0..5 {
            de.step();
        }

        assert!(de.best_fitness > initial_fitness);
    }
} 