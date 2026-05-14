mod common;

use common::fcns::{QuadraticConstraints, QuadraticObjective};
use nalgebra::{SMatrix, U1, U2};

use non_convex_opt::algorithms::limited_memory_bfgs::{
    AdvancedConf, BacktrackingConf, CommonConf, GoldenSectionConf, LBFGSConf, LineSearchConf,
    MemoryAdaptation, NumericalSafeguards, RestartStrategy, StagnationDetection, StrongWolfeConf,
    LBFGS,
};
use non_convex_opt::utils::config::OptConf;
use non_convex_opt::utils::opt_prob::{OptProb, OptimizationAlgorithm};

fn default_advanced() -> AdvancedConf {
    AdvancedConf {
        adaptive_parameters: false,
        adaptation_rate: 0.1,
        restart_strategy: RestartStrategy::None,
        stagnation_detection: StagnationDetection {
            stagnation_window: 50,
            improvement_threshold: 1e-6,
        },
        memory_adaptation: MemoryAdaptation {
            adaptive_memory: false,
            min_memory_size: 5,
            max_memory_size: 20,
        },
        numerical_safeguards: NumericalSafeguards {
            conditioning_threshold: 1e-12,
            curvature_threshold: 1e-8,
        },
        success_history_size: 20,
        improvement_history_size: 20,
    }
}

fn run(line_search: LineSearchConf, steps: usize) -> (f64, [f64; 2]) {
    let conf = LBFGSConf {
        common: CommonConf { memory_size: 10 },
        line_search,
        advanced: default_advanced(),
    };
    let init_x = SMatrix::<f64, 1, 2>::from_row_slice(&[0.5, 0.5]);
    let obj_f = QuadraticObjective { a: 1.0, b: 100.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mut lbfgs: LBFGS<f64, U1, U2> = LBFGS::new(conf, init_x, opt_prob, &OptConf::default(), 0);
    let initial_fitness = lbfgs.st.best_f;
    for _ in 0..steps {
        lbfgs.step();
    }
    (
        lbfgs.st.best_f - initial_fitness,
        [lbfgs.st.best_x[0], lbfgs.st.best_x[1]],
    )
}

#[test]
fn test_backtracking_line_search() {
    let (gain, x) = run(
        LineSearchConf::Backtracking(BacktrackingConf { c1: 1e-4, rho: 0.5 }),
        10,
    );
    assert!(gain > 0.0);
    assert!(x.iter().all(|v| (0.0..=1.0).contains(v)));
}

#[test]
fn test_strong_wolfe_line_search() {
    let (gain, _) = run(
        LineSearchConf::StrongWolfe(StrongWolfeConf {
            c1: 1e-4,
            c2: 0.9,
            max_iters: 100,
        }),
        5,
    );
    assert!(gain > 0.0);
}

#[test]
fn test_golden_section_line_search() {
    let (gain, _) = run(
        LineSearchConf::GoldenSection(GoldenSectionConf {
            tol: 1e-6,
            max_iters: 100,
            bracket_factor: 2.0,
        }),
        5,
    );
    assert!(gain > 0.0);
}
