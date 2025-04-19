use nalgebra::DVector;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum MutationOperator {
    Gaussian(Gaussian),
    Uniform(Uniform),
    NonUniform(NonUniform),
    Polynomial(Polynomial),
}

impl MutationOperator {
    pub fn mutate<T: FloatNum>(&self, individual: &DVector<T>, bounds: (T, T), generation: usize) -> DVector<T> {
        match self {
            MutationOperator::Gaussian(mutation) => mutation.mutate(individual, bounds),
            MutationOperator::Uniform(mutation) => mutation.mutate(individual, bounds),
            MutationOperator::NonUniform(mutation) => mutation.mutate(individual, bounds, generation),
            MutationOperator::Polynomial(mutation) => mutation.mutate(individual, bounds),
        }
    }
}

pub struct Gaussian {
    pub mutation_rate: f64,
    pub sigma: f64,
}

impl Gaussian {
    pub fn new(mutation_rate: f64, sigma: f64) -> Self {
        Self { mutation_rate, sigma }
    }

    pub fn mutate<T: FloatNum>(&self, individual: &DVector<T>, bounds: (T, T)) -> DVector<T> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, self.sigma).unwrap();
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                let noise = T::from_f64(normal.sample(&mut rng)).unwrap();
                mutated[i] = (mutated[i] + noise).clamp(bounds.0, bounds.1);
            }
        }
        mutated
    }
}

pub struct Uniform {
    pub mutation_rate: f64,
}

impl Uniform {
    pub fn new(mutation_rate: f64) -> Self {
        Self { mutation_rate }
    }

    pub fn mutate<T: FloatNum>(&self, individual: &DVector<T>, bounds: (T, T)) -> DVector<T> {
        let mut rng = rand::rng();
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                mutated[i] = T::from_f64(
                    rng.random_range(bounds.0.to_f64().unwrap()..bounds.1.to_f64().unwrap())
                ).unwrap();
            }
        }
        mutated
    }
}

pub struct NonUniform {
    pub mutation_rate: f64,
    pub b: f64,  // Shape parameter
    pub max_generations: usize,
}

impl NonUniform {
    pub fn new(mutation_rate: f64, b: f64, max_generations: usize) -> Self {
        Self { mutation_rate, b, max_generations }
    }

    pub fn mutate<T: FloatNum>(&self, individual: &DVector<T>, bounds: (T, T), generation: usize) -> DVector<T> {
        let mut rng = rand::rng();
        let mut mutated = individual.clone();
        let r = T::from_f64(generation as f64 / self.max_generations as f64).unwrap();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                let delta = if rng.random_bool(0.5) {
                    bounds.1 - mutated[i]
                } else {
                    mutated[i] - bounds.0
                };
                
                let power = T::from_f64(
                    (T::one() - r).to_f64().unwrap().powf(self.b) * rng.random::<f64>()
                ).unwrap();
                
                if rng.random_bool(0.5) {
                    mutated[i] += delta * power;
                } else {
                    mutated[i] -= delta * power;
                }
                
                mutated[i] = mutated[i].clamp(bounds.0, bounds.1);
            }
        }
        mutated
    }
}

pub struct Polynomial {
    pub mutation_rate: f64,
    pub eta_m: f64,  // Distribution index
}

impl Polynomial {
    pub fn new(mutation_rate: f64, eta_m: f64) -> Self {
        Self { mutation_rate, eta_m }
    }

    pub fn mutate<T: FloatNum>(&self, individual: &DVector<T>, bounds: (T, T)) -> DVector<T> {
        let mut rng = rand::rng();
        let mut mutated = individual.clone();

        for i in 0..individual.len() {
            if rng.random::<f64>() < self.mutation_rate {
                let r = rng.random::<f64>();
                let delta = if r < 0.5 {
                    (2.0 * r).powf(1.0 / (self.eta_m + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - r)).powf(1.0 / (self.eta_m + 1.0))
                };
                
                mutated[i] = (mutated[i] + T::from_f64(delta).unwrap() * 
                    (bounds.1 - bounds.0)).clamp(bounds.0, bounds.1);
            }
        }
        mutated
    }
} 