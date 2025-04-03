use rand::Rng;
use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction};

pub struct AutoMALA<T: FloatNum, F: ObjectiveFunction<T>> {
    pub k: T,
    pub step_size: DMatrix<T>,
    pub obj: F,
}

impl<T: FloatNum, F: ObjectiveFunction<T>> AutoMALA<T> {
    pub fn new(obj: F) -> Self {
        let k = T::from_f64(1.38064852e-23).unwrap(); // Boltzmann constant
        let dimension = obj.x_upper_bound().as_ref().map_or(1, |b| b.len());
        let step_size = DMatrix::identity(dimension, dimension);

        AutoMALA { k, step_size, obj }
    }

    fn project(&self, x: &DVector<T>) -> DVector<T> {
        if let (Some(x_ub), Some(x_lb)) = (&self.obj.x_upper_bound(), &self.obj.x_lower_bound()) {
            x.component_mul(&(x_ub.clone() - x_lb.clone())) + x_lb.clone()
        } else {
            x.clone()
        }
    }

    pub fn accept_reject(
        &self,
        x_old: &DVector<T>,
        x_new: &DVector<T>,
        p_old: &DVector<T>,
        p_new: &DVector<T>,
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
        let delta_f = self.obj.f(self.project(x_new)) - self.obj.f(self.project(x_old));

        let r: T;
        if t_swap > 0.0 { // Pass in next temperature to signal global move. 
            let delta_t = T::from_f64((1.0 / t) - (1.0 / t_swap)).unwrap();
            r = (delta_f / (self.k * delta_t * delta_x)).exp();
        } else { // Pass in negative anything to signal local move. 
            let delta_p = (p_new.iter().map(|&val| val * val).sum::<T>() - p_old.iter().map(|&val| val * val).sum::<T>()) / T::from_f64(2.0).unwrap();
            r = ((delta_f - delta_p) / (self.k * delta_x * T::from_f64(t).unwrap())).exp();
        }

        let mut rng = rand::rng(); 
        let u = T::from_f64(rng.random::<f64>()).unwrap();
        u < r
    }

    pub fn local_move(
        &self,
        x_old: &DVector<T>,
    ) -> (DVector<T>, DVector<T>,) {
        let mut rng = rand::rng(); 
    }
        
