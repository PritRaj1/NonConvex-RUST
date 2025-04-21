use nalgebra::{DVector, DMatrix};
use rand::Rng;
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum SelectionOperator {
    RouletteWheel(RouletteWheel), // A.K.A. Proportional Selection
    Tournament(Tournament),
    Residual(Residual), // A.K.A. Stochastic Remainder Selection
}

impl SelectionOperator {
    pub fn select<T: FloatNum>(&self, population: &DMatrix<T>, fitness: &DVector<T>, constraint: &DVector<bool>) -> DMatrix<T> {
        match self {
            SelectionOperator::RouletteWheel(selection) => selection.select(population, fitness, constraint),
            SelectionOperator::Tournament(selection) => selection.select(population, fitness, constraint),
            SelectionOperator::Residual(selection) => selection.select(population, fitness, constraint),
        }
    }
}

pub struct RouletteWheel { 
    pub population_size: usize,
    pub num_parents: usize,
}

impl RouletteWheel {
    pub fn new(population_size: usize, num_parents: usize) -> Self {
        RouletteWheel { population_size, num_parents }
    }

    pub fn select<T: FloatNum>(
        &self, 
        population: &DMatrix<T>, 
        fitness: &DVector<T>, 
        constraint: &DVector<bool>
    ) -> DMatrix<T> {
        
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

pub struct Tournament {
    pub population_size: usize,
    pub num_parents: usize,
    pub tournament_size: usize,
}

impl Tournament {
    pub fn new(population_size: usize, num_parents: usize, tournament_size: usize) -> Self {
        Tournament { population_size, num_parents, tournament_size }
    }

    pub fn select<T: FloatNum>(
        &self, 
        population: &DMatrix<T>, 
        fitness: &DVector<T>, 
        constraint: &DVector<bool>
    ) -> DMatrix<T> {
        let mut selected = DMatrix::<T>::zeros(self.num_parents, population.ncols());
        let mut rng = rand::rng(); 

        for i in 0..self.num_parents {
            let mut tournament_indices = Vec::new();

            // Randomly select `tournament_size` valid individuals
            while tournament_indices.len() < self.tournament_size {
                let index = rng.random_range(0..population.nrows());
                if constraint[index] { // Only add if it satisfies the constraint
                    tournament_indices.push(index);
                }
            }

            let best_index = *tournament_indices.iter()
                .max_by(|&&a, &&b| fitness[a].partial_cmp(&fitness[b]).unwrap())
                .unwrap();

            selected.set_row(i, &population.row(best_index));
        }

        selected
    }
}

pub struct Residual { 
    pub population_size: usize,
    pub num_parents: usize,
}

impl Residual {
    pub fn new(population_size: usize, num_parents: usize) -> Self {
        Residual { population_size, num_parents }
    }

    pub fn select<T: FloatNum>(
        &self, 
        population: &DMatrix<T>, 
        fitness: &DVector<T>, 
        constraint: &DVector<bool>
    ) -> DMatrix<T> {
        let mut selected = DMatrix::<T>::zeros(self.num_parents, population.ncols());
        let mut rng = rand::rng();

        // Calculate expected values
        let sum = fitness.iter()
            .zip(constraint.iter())
            .filter(|(_, &valid)| valid)
            .fold(T::zero(), |acc, (&x, _)| acc + x);
        
        // Normalize and scale fitness to prevent very small residuals
        let scale = T::from_f64(self.num_parents as f64).unwrap();
        let mut expected_values = vec![T::zero(); fitness.len()];
        let mut residual_probabilities = vec![T::zero(); fitness.len()];
        let mut guaranteed_count = 0;
        let mut remaining_indices = Vec::new();

        // Calculate expected values and residuals
        for (j, (&fit, &valid)) in fitness.iter().zip(constraint.iter()).enumerate() {
            if valid {
                let expected = (fit / sum) * scale;
                let int_part = expected.floor();
                expected_values[j] = int_part;
                residual_probabilities[j] = expected - int_part;
                
                guaranteed_count += int_part.to_usize().unwrap_or(0);
                if residual_probabilities[j] > T::zero() {
                    remaining_indices.push(j);
                }
            }
        }

        // Deterministic selections first (integer replication)
        let mut parent_index = 0;
        for j in 0..fitness.len() {
            for _ in 0..expected_values[j].to_usize().unwrap_or(0) {
                if parent_index < self.num_parents {
                    selected.set_row(parent_index, &population.row(j));
                    parent_index += 1;
                }
            }
        }

        // Stochastic remainder selections - select based on residual probabilities
        let mut remaining_spots = self.num_parents - parent_index;
        
        // If no remaining indices but spots left, add all valid individuals
        if remaining_indices.is_empty() && remaining_spots > 0 {
            remaining_indices = (0..population.nrows())
                .filter(|&i| constraint[i])
                .collect();
        }

        while remaining_spots > 0 && !remaining_indices.is_empty() {
            let total_residual = remaining_indices.iter()
                .fold(T::zero(), |acc, &i| acc + residual_probabilities[i]);

            if total_residual <= T::zero() {
                // If all residuals are zero, select randomly
                let idx = remaining_indices[rng.random_range(0..remaining_indices.len())];
                selected.set_row(parent_index, &population.row(idx));
                parent_index += 1;
                remaining_spots -= 1;
                
                // Remove selected index
                if let Some(pos) = remaining_indices.iter().position(|&x| x == idx) {
                    remaining_indices.swap_remove(pos);
                }
            } else {
                let r = rng.random::<f64>();
                let mut cumsum = T::zero();
                
                for &idx in &remaining_indices {
                    cumsum += residual_probabilities[idx] / total_residual;
                    if T::from_f64(r).unwrap() <= cumsum {
                        selected.set_row(parent_index, &population.row(idx));
                        parent_index += 1;
                        remaining_spots -= 1;
                        
                        // Remove selected index and reset its probability
                        if let Some(pos) = remaining_indices.iter().position(|&x| x == idx) {
                            remaining_indices.swap_remove(pos);
                        }
                        residual_probabilities[idx] = T::zero();
                        break;
                    }
                }
            }
        }

        selected
    }
}
