mod common;

use crate::common::fcns::{
    QuadraticConstraints, QuadraticObjective, RosenbrockConstraints, RosenbrockObjective,
};
use nalgebra::{DMatrix, DVector};
use non_convex_opt::algorithms::parallel_tempering::{
    AdaptiveCovariance, AutoConf, FitnessWeightedCovariance, MetropolisHastings, Preconditioner,
    SampleCovariance, ShrinkageCovariance, UpdateConf, PT,
};
use non_convex_opt::utils::config::OptConf;
use non_convex_opt::utils::{
    config::{AlgConf, Config},
    opt_prob::{OptProb, OptimizationAlgorithm}
};

#[test]
fn test_metropolis_hastings_accept_reject() {
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    let x_old = DVector::from_vec(vec![0.1, 0.1]); // High Rosenbrock value
    let x_new = DVector::from_vec(vec![0.95, 0.95]); // Lower Rosenbrock value, but closer to optimum

    let mut mh: MetropolisHastings<f64, nalgebra::Dyn> =
        MetropolisHastings::new(opt_prob, &UpdateConf::Auto(AutoConf {}), x_old.clone(), 42);
    let constraints_new = true;

    let x_better = DVector::from_vec(vec![0.95, 0.9025]); // Closer to Rosenbrock optimum [1,1]
    let accepted_uphill = mh.accept_reject(&x_old, &x_better, constraints_new, 0.5);
    assert!(accepted_uphill);

    let accepted_constrained = mh.accept_reject(&x_old, &x_new, false, 0.5);
    assert!(!accepted_constrained);
}

#[test]
fn test_pt_step_runs() {
    let conf = Config::new(include_str!("jsons/pt.json")).unwrap();
    let pt_conf = match conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf")
    };

    let init_pop = DMatrix::from_vec(2, 2, vec![0.5, 0.5, 0.5, 0.5]);
    let obj_f = RosenbrockObjective { a: 1.0, b: 1.0 };
    let constraints = RosenbrockConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));
    let mut pt = PT::new(
        pt_conf,
        init_pop,
        opt_prob,
        &OptConf {
            max_iter: 5,
            ..OptConf::default()
        },
        42,
    );

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
}

fn create_test_pt_pcn() -> PT<f64, nalgebra::Dyn, nalgebra::Dyn> {
    let conf = Config::new(include_str!("jsons/pt_pcn.json")).unwrap();
    let pt_conf = match conf.alg_conf {
        AlgConf::PT(pt_conf) => pt_conf,
        _ => panic!("Expected PTConf")
    };

    // Need fairly large pop for covariance to change
    let init_pop = DMatrix::from_vec(
        10,
        2,
        vec![
            0.90, 0.90, // 0.81
            0.95, 0.80, // 0.76
            0.88, 0.86, // 0.7568
            0.92, 0.85, // 0.782
            0.89, 0.95, // 0.8455
            0.97, 0.78, // 0.7566
            0.90, 0.86, // 0.774
            0.88, 0.90, // 0.792
            0.93, 0.90, // 0.837
            0.91, 0.91, // 0.8281
        ],
    );
    let obj_f = QuadraticObjective { a: 1.0, b: 1.0 };
    let constraints = QuadraticConstraints {};
    let opt_prob = OptProb::new(Box::new(obj_f), Some(Box::new(constraints)));

    PT::new(
        pt_conf,
        init_pop,
        opt_prob,
        &OptConf {
            max_iter: 20,
            ..OptConf::default()
        },
        42,
    )
}

#[test]
fn test_sample_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.001));
    pt.set_preconditioner(preconditioner);

    assert_eq!(pt.covariance_matrices.len(), pt.get_num_replicas());

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        for i in 0..cov.nrows() {
            assert!(cov[(i, i)] > 0.0, "Diagonal element should be positive");
        }

        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < 1e-10,
                    "Matrix should be symmetric"
                );
            }
        }
    }
}

#[test]
fn test_fitness_weighted_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(FitnessWeightedCovariance::new(0.01, 0.5));
    pt.set_preconditioner(preconditioner);

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Covariance matrix should be positive definite"
        );

        for i in 0..cov.nrows() {
            assert!(
                cov[(i, i)] >= 0.009,
                "Diagonal should include regularization"
            );
        }
    }
}

#[test]
fn test_adaptive_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(AdaptiveCovariance::new(0.01, 0.1, 0.234));
    pt.set_preconditioner(preconditioner);

    let _initial_trace: f64 = pt.covariance_matrices[0].trace();

    for _ in 0..10 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Covariance matrix should be positive definite"
        );

        let condition_number = {
            let eigenvalues = cov.symmetric_eigenvalues();
            let max_eig = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_eig = eigenvalues
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b.max(1e-12)));
            max_eig / min_eig
        };
        assert!(
            condition_number < 1e6,
            "Condition number should be reasonable"
        );
    }
}

#[test]
fn test_shrinkage_covariance_preconditioner() {
    let mut pt = create_test_pt_pcn();
    let initial_best = pt.st.best_f;

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(ShrinkageCovariance::new(0.3));
    pt.set_preconditioner(preconditioner);

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f >= initial_best || (pt.st.best_f - initial_best).abs() < 1e-10);

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Covariance matrix should be positive definite"
        );

        let trace = cov.trace();
        let avg_diagonal = trace / cov.nrows() as f64;

        for i in 0..cov.nrows() {
            for j in 0..cov.ncols() {
                if i != j {
                    assert!(
                        cov[(i, j)].abs() <= avg_diagonal,
                        "Off-diagonal elements should be shrunk"
                    );
                }
            }
        }
    }
}

#[test]
fn test_preconditioner_covariance_update() {
    let mut pt = create_test_pt_pcn();

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.0001));
    pt.set_preconditioner(preconditioner);

    let initial_covariances: Vec<_> = pt.covariance_matrices.clone();

    for _ in 0..25 {
        pt.step();
    }

    let updated = pt
        .covariance_matrices
        .iter()
        .zip(initial_covariances.iter())
        .any(|(new_cov, old_cov)| {
            for i in 0..new_cov.nrows() {
                for j in 0..new_cov.ncols() {
                    if (new_cov[(i, j)] - old_cov[(i, j)]).abs() > 1e-12 {
                        return true;
                    }
                }
            }
            false
        });

    assert!(
        updated,
        "Covariance matrices should be updated after sufficient iterations"
    );
}

#[test]
fn test_pcn_variance_parameter_decrease() {
    let mut pt = create_test_pt_pcn();

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.01));
    pt.set_preconditioner(preconditioner);

    let initial_best = pt.st.best_f;

    for _ in 0..15 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());
    assert!(pt.st.best_f > -1e6, "Algorithm should not diverge");

    assert!(
        pt.st.best_f >= initial_best - 1.0,
        "Algorithm should not deteriorate significantly"
    );
}

#[test]
fn test_preconditioner_with_infeasible_individuals() {
    let mut pt = create_test_pt_pcn();

    for replica_idx in 0..pt.get_num_replicas() {
        pt.replicas[replica_idx].population[(0, 0)] = -1.0; // Outside QuadraticConstraints bounds
        pt.replicas[replica_idx].population[(0, 1)] = 2.0; // Outside QuadraticConstraints bounds
        pt.replicas[replica_idx].constraints[0] = false;
    }

    let preconditioner: Box<dyn Preconditioner<f64, nalgebra::Dyn, nalgebra::Dyn> + Send + Sync> =
        Box::new(SampleCovariance::new(0.01));
    pt.set_preconditioner(preconditioner);

    for _ in 0..5 {
        pt.step();
    }

    assert!(pt.st.best_f.is_finite());

    for cov in &pt.covariance_matrices {
        assert!(
            cov.determinant() > 0.0,
            "Should compute valid covariance despite infeasible individuals"
        );
    }
}
