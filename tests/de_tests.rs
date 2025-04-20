mod common;
use nalgebra::DMatrix;
use non_convex_opt::differential_evolution::de::DE;
use non_convex_opt::utils::config::DEConf;
use non_convex_opt::utils::alg_conf::de_conf::DEStrategy;
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{RosenbrockObjective, RosenbrockConstraints};

#[test]
fn test_de_basic() {
    let conf = DEConf {
        population_size: 50,
        f: 0.8,
        cr: 0.9,
        strategy: DEStrategy::Best2Bin,  // Changed to Best2Bin which often performs better
    };

    // Initialize population in a better range for Rosenbrock
    let mut init_pop = DMatrix::zeros(50, 2);
    for i in 0..50 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 2.0 - 1.0; // Range [-1, 1]
        }
    }

    let obj_f = RosenbrockObjective{ a: 1.0, b: 100.0 };
    let constraints = RosenbrockConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut de = DE::new(conf, init_pop.clone(), opt_prob);
    let initial_fitness = de.best_fitness;
    
    // Run for more iterations to ensure improvement
    for _ in 0..50 {
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
                init_pop[(i, j)] = rand::random::<f64>() * 2.0 - 1.0; // Range [-1, 1]
            }
        }

        let obj_f = RosenbrockObjective{ a: 1.0, b: 100.0 };
        let constraints = RosenbrockConstraints{};
        let opt_prob = OptProb::new(obj_f, Some(constraints));
        
        let mut de = DE::new(conf, init_pop.clone(), opt_prob);
        let initial_fitness = de.best_fitness;
        
        // Run for more iterations
        for _ in 0..50 {
            de.step();
        }

        assert!(de.best_fitness > initial_fitness, 
            "Strategy {:?} failed to improve fitness", strategy);
    }
} 