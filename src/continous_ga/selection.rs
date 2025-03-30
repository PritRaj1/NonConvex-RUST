use nalgebra::{DVector, DMatrix};
use rand::Rng;
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum SelectionOperator {
    RouletteWheel,
    Tournament,
    SRS,
}

pub struct RouletteWheel {
    pub population_size: usize,
    pub num_parents: usize,
}

impl RouletteWheel {
    pub fn new(population_size: usize, num_parents: usize) -> Self {
        RouletteWheel { population_size, num_parents }
    }

    pub fn select<T: FloatNum>(&self, population: &DMatrix<T>, fitness: &DVector<T>, constraint: &DVector<bool>) -> DMatrix<T> {
        
        // Normalized selection probabilities only for valid individuals
        let sum = fitness.iter()
            .zip(constraint.iter())
            .filter(|(_, &valid)| valid)
            .fold(T::zero(), |acc, (&x, _)| acc + x);

        let mut llhoods = DVector::<T>::zeros(fitness.len());
        for (j, (&fit, &valid)) in fitness.iter().zip(constraint.iter()).enumerate() {
            if valid {
                llhoods[j] = fit / sum;
            }
        }
        
        let mut selected = DMatrix::<T>::zeros(self.num_parents, population.ncols());
        let mut rng = rand::rng();

        for i in 0..self.num_parents {
            let mut r = T::from_f64(rng.random_range(0.0..1.0));
            for j in 0..population.nrows() {
                if !constraint[j] {
                    continue; // Skip individuals that don't satisfy constraints
                }

                r = Some(r.unwrap_or(T::zero()) - llhoods[j]);
                if r <= Some(T::zero()) {
                    selected.set_row(i, &population.row(j));
                    break;
                }
            }
        }
        selected
    }
}