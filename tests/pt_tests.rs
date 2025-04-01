use non_convex_opt::parallel_tempering::metropolis_hastings::{MetropolisHastings};
use non_convex_opt::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction, BooleanConstraintFunction, OptProb};
use nalgebra::{DVector};

#[test]
fn test_metropolis_hastings_accept_reject() {
    let x_bounds = vec![0.0, 1.0]; // Example bounds for a 2D problem
    let mh = MetropolisHastings::new(x_bounds);

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = DVector::from_vec(vec![0.6, 0.6]);
    let f_old = 1.0;
    let f_new = 1.2;
    let constraints_new = true; // Assume new solution satisfies constraints
    let t = 1.0;
    let t_swap = 2.0;

    let accepted = mh.accept_reject(&x_old, &x_new, f_old, f_new, constraints_new, t, t_swap);
    
    assert!(accepted);
}

#[test]
fn test_metropolis_hastings_local_move() {
    let x_bounds = vec![0.0, 1.0]; // Example bounds for a 2D problem
    let mh = MetropolisHastings::new(x_bounds);

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = mh.local_move(&x_old);

    assert!(x_new.iter().all(|x| x >= &0.0 && x <= &1.0));
}

#[test]
fn test_metropolis_hastings_update_step_size() {
    let x_bounds = vec![0.0, 1.0]; // Example bounds for a 2D problem
    let mut mh = MetropolisHastings::new(x_bounds);

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = DVector::from_vec(vec![0.6, 0.6]);
    let alpha = 0.1;
    let omega = 0.1;
    mh.update_step_size(&x_old, &x_new, alpha, omega);

    assert!(mh.step_size[(0, 0)] == 0.1);
    assert!(mh.step_size[(1, 1)] == 0.1);
}

