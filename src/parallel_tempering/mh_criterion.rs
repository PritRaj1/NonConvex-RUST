use rand::Rng;
use nalgebra::DVector;
use crate::utils::opt_prob::FloatNumber as FloatNum;
use crate::utils::opt_prob::ObjectiveFunction;

pub enum MHCriterion<T: FloatNum> {
    MetropolisHastings(MetropolisHastings<T>),
    // MA_Langevin(MA_Langevin<T>),
}

pub struct MetropolisHastings<T: FloatNum> {
    pub k: T,
    pub x_bounds: Vec<T>, // lower and upper bounds for each dimension
}

impl<T: FloatNum> MetropolisHastings<T> {
    pub fn new(x_bounds: Vec<T>) -> Self {
        let k = T::from_f64(1.38064852e-23).unwrap(); // Boltzmann constant
        MetropolisHastings { k, x_bounds }
    }

    pub fn accept_reject(
        &self, 
        x_old: &DVector<T>,
        x_new: &DVector<T>,
        f_old: T,
        f_new: T,
        constraints_new: bool,
        t: f64,
        t_swap: f64, 
    ) -> bool {

        // Reject if new solution violates constraints
        if !constraints_new {
            return false; 
        }

        let diff = x_new - x_old;
        let delta_x = diff.dot(&diff).sqrt();
        let delta_f = f_new - f_old;

        let r: T;
        if t_swap > 0.0 { // Pass in next temperature to signal global move. 
            let delta_t = T::from_f64((1.0 / t) - (1.0 / t_swap)).unwrap();
            r = (delta_f / (self.k * delta_t * delta_x)).exp();
        } else { // Pass in negative anything to signal local move. 
            r = (delta_f / (self.k * delta_x * T::from_f64(t).unwrap())).exp();
        }

        let mut rng = rand::rng(); 
        let u = T::from_f64(rng.random::<f64>()).unwrap();
        u < r
    }
}

