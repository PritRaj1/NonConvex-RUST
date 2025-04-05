use rand::Rng;
use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction};

pub struct MetropolisHastings<T: FloatNum, F: ObjectiveFunction<T>> {
    pub k: T,
    pub step_size: DMatrix<T>,
    pub mala_step_size: T,
    pub obj: F,
}

impl<T: FloatNum, F: ObjectiveFunction<T>> MetropolisHastings<T, F> {
    pub fn new(obj: F, step_size_scalar: T) -> Self {
        let k = T::from_f64(1.38064852e-23).unwrap(); // Boltzmann constant
        let dimension = obj.x_upper_bound().as_ref().map_or(1, |b| b.len());
        let step_size = DMatrix::identity(dimension, dimension) * step_size_scalar;
        let mala_step_size = step_size_scalar;

        MetropolisHastings { k, step_size, mala_step_size, obj }  
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
        let delta_f = self.obj.f(&self.project(x_new)) - self.obj.f(&self.project(x_old));

        let r: T;
        if t_swap > 0.0 { // Pass in next temperature to signal global move
            let delta_t = T::from_f64((1.0 / t) - (1.0 / t_swap)).unwrap();
            r = (delta_f / (self.k * delta_t * delta_x)).exp();
        } else { // Pass in negative anything to signal local move
            r = (delta_f / (self.k * delta_x * T::from_f64(t).unwrap())).exp();
        }

        let mut rng = rand::rng(); 
        let u = T::from_f64(rng.random::<f64>()).unwrap();
        u < r
    }

    pub fn local_move(
        &self,
        x_old: &DVector<T>,
    ) -> DVector<T> {
        let mut rng = rand::rng();
        let mut x_new = x_old.clone();
        let random_vec = DVector::from_fn(x_old.len(), |_, _| T::from_f64(rng.random::<f64>()).unwrap());

        if let Some(grad) = self.obj.gradient(&x_old) {
            // Use Metropolis Adjusted Langevin Algorithm (MALA)
            x_new += grad * self.mala_step_size + random_vec.component_mul(&self.step_size.diagonal());
        } else {
            // Use standard Metropolis Hastings
            x_new += random_vec.component_mul(&self.step_size.diagonal());
        }

        x_new
    }

    pub fn update_step_size(
        &mut self,
        x_old: &DVector<T>,
        x_new: &DVector<T>,
        alpha: T,
        omega: T,
    ) {
        let r = DVector::from_fn(x_old.len(), |i, _| (x_new[i] - x_old[i]).abs());
        for i in 0..x_old.len() {
            self.step_size[(i, i)] = (T::from_f64(1.0).unwrap() - alpha) * self.step_size[(i, i)] + alpha * omega * r[i];
        }
    }
}
