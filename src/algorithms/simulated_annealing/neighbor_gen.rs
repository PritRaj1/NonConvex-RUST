use nalgebra::DVector;
use rand::Rng;
use rand_distr::{Normal, StandardNormal};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use rayon::prelude::*;

pub enum MoveType {
    RandomDrift,
    MALA,
}

pub struct GaussianGenerator<T: FloatNum> {
    pub move_type: MoveType,
    pub prob: OptProb<T>,
    pub mala_step_size: T,
}

impl<T: FloatNum> GaussianGenerator<T> {
    pub fn new(prob: OptProb<T>, mala_step_size: T) -> Self {
        let move_type = if prob.objective.gradient(&DVector::zeros(1)).is_some() {
            MoveType::MALA
        } else {
            MoveType::RandomDrift
        };

        Self { move_type, prob, mala_step_size }
    }

    pub fn generate(&self, current: &DVector<T>, step_size: f64, bounds: (T, T), temperature: T) -> DVector<T> {
        match self.move_type {
            MoveType::RandomDrift => self.random_drift(current, step_size, bounds),
            MoveType::MALA => self.mala_move(current, temperature, bounds),
        }
    }

    fn random_drift(&self, current: &DVector<T>, step_size: f64, bounds: (T, T)) -> DVector<T> {
        let mut neighbor = current.clone();
        
        neighbor.as_mut_slice()
            .par_chunks_mut(1)
            .enumerate()
            .for_each(|(i, val)| {
                let mut rng = rand::rng();
                let step = T::from_f64(
                    rng.sample::<f64, _>(Normal::new(0.0, step_size).unwrap())
                ).unwrap();
                val[0] = (current[i] + step).clamp(bounds.0, bounds.1);
            });
        
        neighbor
    }

    fn mala_move(&self, current: &DVector<T>, temperature: T, bounds: (T, T)) -> DVector<T> {
        let mut rng = rand::rng();
        let step = self.mala_step_size * temperature;
        
        let grad = self.prob.objective.gradient(current).unwrap();
        let drift = grad * step;
        
        let noise = DVector::from_fn(current.len(), |_, _| {
            T::from_f64(rng.sample::<f64, _>(StandardNormal)).unwrap()
        }) * (step * T::from_f64(2.0).unwrap()).sqrt();
        
        let mut new_pos = current + drift + noise;
        for i in 0..new_pos.len() {
            new_pos[i] = new_pos[i].clamp(bounds.0, bounds.1); // Not valid without Jacobian, but hey!
        }
        
        new_pos
    }
} 