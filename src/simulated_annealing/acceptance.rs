use rand::Rng;
use nalgebra::DVector;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};

pub enum AcceptanceType {
    Metropolis,
    MALA,
}

pub trait AcceptanceCriterion<T: FloatNum> {
    fn accept(
        &self, 
        current_x: &DVector<T>,
        current_fitness: T,
        new_x: &DVector<T>,
        new_fitness: T,
        temperature: T,
        step_size: T,
    ) -> bool;
}

pub struct MetropolisAcceptance<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub acceptance_type: AcceptanceType,
    pub prob: OptProb<T, F, G>,
    k: T,  // Boltzmann constant
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> MetropolisAcceptance<T, F, G> {
    pub fn new(prob: OptProb<T, F, G>) -> Self {
        let acceptance_type = if prob.objective.gradient(&DVector::zeros(1)).is_some() {
            AcceptanceType::MALA
        } else {
            AcceptanceType::Metropolis
        };

        Self {
            acceptance_type,
            prob,
            k: T::from_f64(1.380649e-23).unwrap(),
        }
    }

    pub fn accept(
        &self,
        current_x: &DVector<T>,
        current_fitness: T,
        new_x: &DVector<T>,
        new_fitness: T,
        temperature: T,
        step_size: T,
    ) -> bool {
        let mut rng = rand::rng();

        if new_fitness > current_fitness {
            return true;
        }

        let diff = new_x - current_x;
        let delta_x = diff.dot(&diff).sqrt();
        let delta_f = new_fitness - current_fitness;

        let r = match self.acceptance_type {
            AcceptanceType::Metropolis => {
                (delta_f / (temperature * delta_x * self.k)).exp()
            },
            AcceptanceType::MALA => {
                let grad = self.prob.objective.gradient(current_x).unwrap();
                let proposal_grad = self.prob.objective.gradient(new_x).unwrap();
                
                let langevin_correction = -(
                    (new_x - current_x - grad.clone() * step_size * temperature)
                        .dot(&(new_x - current_x - grad * step_size * temperature))
                    / (T::from_f64(4.0).unwrap() * step_size * temperature)
                ) + (
                    (current_x - new_x - proposal_grad.clone() * step_size * temperature)
                        .dot(&(current_x - new_x - proposal_grad * step_size * temperature))
                    / (T::from_f64(4.0).unwrap() * step_size * temperature)
                );

                (delta_f / (self.k * temperature * delta_x) + langevin_correction).exp()
            }
        };

        rng.random::<f64>() < r.to_f64().unwrap()
    }
} 