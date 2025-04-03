use non_convex_opt::parallel_tempering::metropolis_hastings::{MetropolisHastings};
use non_convex_opt::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction, BooleanConstraintFunction, OptProb};
use nalgebra::{DVector};

pub struct Rosenbrock<T: FloatNum> {
    pub a: T,
    pub b: T,
}

impl<T: FloatNum> Rosenbrock<T> {
    pub fn new(a: T, b: T) -> Self {
        Self { a, b }
    }
}

impl<T: FloatNum> ObjectiveFunction<T> for Rosenbrock<T> {
    fn f(&self, x: &DVector<T>) -> T {
        let n = x.len();
        let mut sum = T::zero();
        for i in 0..n-1 {
            sum += self.b * (x[i+1] - x[i].powi(2)).powi(2) + 
                   (self.a - x[i]).powi(2);
        }
        sum
    }
}

impl<T: FloatNum> BooleanConstraintFunction<T> for Rosenbrock<T> {
    fn g(&self, x: &DVector<T>) -> DVector<bool> {
        DVector::from_vec(vec![true; x.len()])
    }
}

impl<T: FloatNum> OptProb<T> for Rosenbrock<T> {
    fn objective(&self, x: &DVector<T>) -> T {
        self.f(x)
    }       

    fn constraints(&self, x: &DVector<T>) -> DVector<bool> {
        self.g(x)
    }
}

#[test]
fn test_metropolis_hastings_accept_reject() {
    let x_bounds = vec![0.0, 1.0];
    let obj_f = Rosenbrock::new(1.0, 100.0);
    let mh = MetropolisHastings::new(x_bounds, obj_f);

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = DVector::from_vec(vec![0.6, 0.6]);
    let constraints_new = true;
    let t = 1.0;
    let t_swap = 2.0;

    let accepted = mh.accept_reject(&x_old, &x_new, constraints_new, t, t_swap);
    
    assert!(accepted);
}

#[test]
fn test_metropolis_hastings_local_move() {
    let x_bounds = vec![0.0, 1.0]; 
    let obj_f = Rosenbrock::new(1.0, 100.0);
    let mh = MetropolisHastings::new(x_bounds, obj_f);

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = mh.local_move(&x_old);

    assert_eq!(x_old.len(), x_new.len());
}

#[test]
fn test_metropolis_hastings_update_step_size() {
    let x_bounds = vec![0.0, 1.0]; 
    let obj_f = Rosenbrock::new(1.0, 100.0);
    let mut mh = MetropolisHastings::new(x_bounds, obj_f);

    let x_old = DVector::from_vec(vec![0.5, 0.5]);
    let x_new = DVector::from_vec(vec![0.6, 0.6]);
    let alpha = 0.1;
    let omega = 0.1;
    mh.update_step_size(&x_old, &x_new, alpha, omega);

    assert_eq!(mh.step_size.nrows(), 2);
    assert_eq!(mh.step_size.ncols(), 2);
}

