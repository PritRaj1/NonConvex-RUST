use nalgebra::{DVector, DMatrix};
use rand::Rng;
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum CrossoverOperator {
    Random(Random),
    Heuristic(Heuristic), // A.K.A. Blend Crossover
}

impl CrossoverOperator {
    pub fn crossover<T: FloatNum>(&self, selected: &DMatrix<T>, fitness: &DVector<T>) -> DMatrix<T> {
        match self {
            CrossoverOperator::Random(crossover) => crossover.crossover(selected),
            CrossoverOperator::Heuristic(crossover) => crossover.crossover(selected, fitness),
        }
    }
}
pub struct Random {
    pub crossover_prob: f64,
    pub population_size: usize,
}

impl Random {
    pub fn new(crossover_prob: f64, population_size: usize) -> Self {
        Self { crossover_prob, population_size }
    }

    pub fn crossover<T: FloatNum>(&self, selected: &DMatrix<T>) -> DMatrix<T> {
        let mut rng = rand::rng();
        let mut offspring = DMatrix::<T>::zeros(self.population_size, selected.ncols());

        let num_parents = selected.nrows();
        let mut offspring_count = 0;

        while offspring_count < self.population_size {
            let i = rng.random_range(0..num_parents);
            let j = rng.random_range(0..num_parents);
            
            if i != j && rng.random::<f64>() < self.crossover_prob {
                let parent1 = selected.row(i);
                let parent2 = selected.row(j);

                let mut child = DVector::<T>::zeros(selected.ncols());
                for k in 0..selected.ncols() {
                    child[k] = if rng.random::<f64>() < 0.5 { parent1[k] } else { parent2[k] };
                }

                if offspring_count < self.population_size {
                    offspring.set_row(offspring_count, &child.transpose());
                    offspring_count += 1;
                }
            }
        }

        offspring
    }
}

pub struct Heuristic {
    pub crossover_prob: f64,
    pub population_size: usize,
}

impl Heuristic {
    pub fn new(crossover_prob: f64, population_size: usize) -> Self {
        Self { crossover_prob, population_size }
    }
    
    pub fn crossover<T: FloatNum>(&self, selected: &DMatrix<T>, fitness: &DVector<T>) -> DMatrix<T> {
        let mut rng = rand::rng();
        let mut offspring = DMatrix::<T>::zeros(self.population_size, selected.ncols());

        let num_parents = selected.nrows();
        let mut offspring_count = 0;

        while offspring_count < self.population_size {
            let i = rng.random_range(0..num_parents);
            let j = rng.random_range(0..num_parents);

            if i != j && rng.random::<f64>() < self.crossover_prob {
                let parent1 = selected.row(i);
                let parent2 = selected.row(j);

                let fit1 = fitness[i];
                let fit2 = fitness[j];

                let (better, worse) = if fit1 > fit2 { (parent1, parent2) } else { (parent2, parent1) };
                let mut child = DVector::<T>::zeros(selected.ncols());

                // Perform heuristic crossover by random
                for k in 0..selected.ncols() {
                    let b = T::from_f64(rng.random::<f64>()).unwrap(); // Random factor between 0 and 1
                    child[k] = b * (better[k] - worse[k]) + worse[k]; // p_new = b * (p1 - p2) + p2
                }

                if offspring_count < self.population_size {
                    offspring.set_row(offspring_count, &child.transpose());
                    offspring_count += 1;
                }
            }
        }

        offspring
    }
}
