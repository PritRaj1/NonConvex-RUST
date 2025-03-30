use nalgebra::{DVector, DMatrix};
use rand::Rng;
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum CrossoverOperator {
    Blend,
    Heuristic,
}

pub struct Blend {
    pub crossover_prob: f64,
    pub alpha: f64,
    pub population_size: usize,
}

impl Blend {
    pub fn new(crossover_prob: f64, alpha: f64, population_size: usize) -> Self {
        Self { crossover_prob, alpha, population_size }
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

                let mut child1 = DVector::<T>::zeros(selected.ncols());
                let mut child2 = DVector::<T>::zeros(selected.ncols());

                for k in 0..selected.ncols() {
                    let p1 = parent1[k];
                    let p2 = parent2[k];

                    let min_p = p1.min(p2);
                    let max_p = p1.max(p2);
                    let d = max_p - min_p;

                    let alpha = T::from_f64(self.alpha).unwrap();
                    let lower_bound = min_p - alpha * d;
                    let upper_bound = max_p + alpha * d;

                    let r1 = T::from_f64(rng.random::<f64>()).unwrap();
                    let r2 = T::from_f64(rng.random::<f64>()).unwrap();
                    
                    child1[k] = lower_bound + r1 * (upper_bound - lower_bound);
                    child2[k] = lower_bound + r2 * (upper_bound - lower_bound);
                }

                if offspring_count < self.population_size {
                    offspring.set_row(offspring_count, &child1.transpose());
                    offspring_count += 1;
                }
                if offspring_count < self.population_size {
                    offspring.set_row(offspring_count, &child2.transpose());
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

                // Perform heuristic crossover randome by randome
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
