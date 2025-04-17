mod common;
use non_convex_opt::limited_memory_bfgs::lbfgs::LBFGS;
use non_convex_opt::utils::alg_conf::lbfgs_conf::{LBFGSConf, CommonConf, LineSearchConf, HagerZhangConf};
use non_convex_opt::utils::opt_prob::OptProb;
use common::fcns::{QuadraticObjective, QuadraticConstraints};
use nalgebra::DVector;

#[test]
fn test_lbfgs() {
    let conf = LBFGSConf {
        common: CommonConf {
            memory_size: 10,
        },
        line_search: LineSearchConf::HagerZhang(HagerZhangConf {
            c1: 1e-4,
            c2: 0.9,
            theta: 0.5,
            gamma: 0.5,
            max_iters: 100,
        }),
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut lbfgs = LBFGS::new(conf, init_x.clone(), opt_prob);
    let initial_fitness = lbfgs.best_fitness;
    
    for _ in 0..10 {
        lbfgs.step();
    }

    assert!(lbfgs.best_fitness > initial_fitness);
    assert!(lbfgs.best_x.iter().all(|&x| x >= 0.0 && x <= 1.0));
}

#[test]
fn test_lbfgs_line_search() {
    let conf = LBFGSConf {
        common: CommonConf {
            memory_size: 10,
        },
        line_search: LineSearchConf::HagerZhang(HagerZhangConf {
            c1: 1e-4,
            c2: 0.9,
            theta: 0.5,
            gamma: 0.5,
            max_iters: 100,
        }),
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut lbfgs = LBFGS::new(conf, init_x.clone(), opt_prob);
    
    for _ in 0..5 {
        lbfgs.step();
        assert!(lbfgs.x.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_lbfgs_memory() {
    let conf = LBFGSConf {
        common: CommonConf {
            memory_size: 3,
        },
        line_search: LineSearchConf::HagerZhang(HagerZhangConf {
            c1: 1e-4,
            c2: 0.9,
            theta: 0.5,
            gamma: 0.5,
            max_iters: 100,
        }),
    };

    let init_x = DVector::from_vec(vec![0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints{};
    let opt_prob = OptProb::new(obj_f, Some(constraints));
    
    let mut lbfgs = LBFGS::new(conf, init_x.clone(), opt_prob);
    
    for _ in 0..5 {
        lbfgs.step();
        assert!(lbfgs.best_x.iter().all(|&x| x.is_finite()));
    }
} 