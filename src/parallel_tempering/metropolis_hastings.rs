use rand::Rng;
use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction};

pub enum MoveType {
    RandomDrift,
    MALA,
}

pub struct MetropolisHastings<T: FloatNum, F: ObjectiveFunction<T>> {
    pub k: T,
    pub move_type: MoveType,
    pub obj: F,
    pub mala_step_size: T,
}

impl<T: FloatNum, F: ObjectiveFunction<T>> MetropolisHastings<T, F> {
    pub fn new(obj: F, mala_step_size: T) -> Self {
        let k = T::from_f64(1.38064852e-23).unwrap(); // Boltzmann constant

        let move_type = if obj.gradient(&DVector::zeros(1)).is_some() {
            MoveType::MALA
        } else {
            MoveType::RandomDrift
        };

        MetropolisHastings { k, move_type, obj, mala_step_size }
    }

    pub fn local_move(&self, x_old: &DVector<T>, step_size: &DMatrix<T>) -> DVector<T> {
        match self.move_type {
            MoveType::MALA => {
                let grad = self.obj.gradient(x_old).expect("Gradient should be available for MALA");
                self.local_move_MALA(x_old, &grad)
            }
            MoveType::RandomDrift => self.local_move_RandomDrift(x_old, step_size),
        }
    }

    fn local_move_RandomDrift(&self, x_old: &DVector<T>, step_size: &DMatrix<T>) -> DVector<T> {
        let mut rng = rand::rng();
        let mut x_new = x_old.clone();
        let random_vec = DVector::from_fn(x_old.len(), |_, _| T::from_f64(rng.random::<f64>()).unwrap());
        x_new += random_vec.component_mul(&step_size.diagonal());
        x_new
    }

    fn local_move_MALA(&self, x_old: &DVector<T>, grad: &DVector<T>) -> DVector<T> {
        let mut rng = rand::rng();
        let mut x_new = x_old.clone();
        let random_vec = DVector::from_fn(x_old.len(), |_, _| T::from_f64(rng.random::<f64>()).unwrap());
        x_new += grad * self.mala_step_size + random_vec.map(|val| (T::from_f64(2.0).unwrap() * self.mala_step_size).sqrt() * val);
        x_new
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
}

pub fn update_step_size<T: FloatNum>(
    step_size: &DMatrix<T>,
    x_old: &DVector<T>,
    x_new: &DVector<T>,
    alpha: T,
    omega: T,
) -> DMatrix<T> {
    let r = DVector::from_fn(x_old.len(), |i, _| (x_new[i] - x_old[i]).abs());
    let mut step_size_new = step_size.clone();
    for i in 0..x_old.len() {
        step_size_new[(i, i)] = (T::from_f64(1.0).unwrap() - alpha) * step_size[(i, i)] + alpha * omega * r[i];
    }
    step_size_new
}
