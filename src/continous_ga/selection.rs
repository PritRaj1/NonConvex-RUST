use nalgebra::{DVector, DMatrix};
use rand::Rng;
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum SelectionOperator {
    RouletteWheel,
    Tournament,
    Residual,
}

pub struct RouletteWheel { // A.K.A. Proportional Selection
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
                let index = rng.gen_range(0..population.nrows());
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

pub struct Residual { // A.K.A. Stochastic Remainder Selection without replacement
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

        let sum = fitness.iter()
            .zip(constraint.iter())
            .filter(|(_, &valid)| valid)
            .fold(T::zero(), |acc, (&x, _)| acc + x);

        let mut expected_integer_copies = vec![0; fitness.len()];
        let mut residual_probabilities = vec![T::zero(); fitness.len()];
        let mut guaranteed_count = 0;
        let mut remaining_indices = Vec::new();

        // Calculate expected integer copies and residual probabilities
        for (j, (&fit, &valid)) in fitness.iter().zip(constraint.iter()).enumerate() {
            if valid {
                let expected = (fit / sum) * T::from_usize(self.num_parents).unwrap();
                let int_part = expected.floor().to_usize().unwrap();
                expected_integer_copies[j] = int_part;
                residual_probabilities[j] = expected - T::from_usize(int_part).unwrap();
                
                guaranteed_count += int_part;
                remaining_indices.push(j); // Keep track of eligible individuals
            }
        }

        // Deterministic selections
        let mut parent_index = 0;
        for j in 0..fitness.len() {
            for _ in 0..expected_integer_copies[j] {
                if parent_index < self.num_parents {
                    selected.set_row(parent_index, &population.row(j));
                    parent_index += 1;
                }
            }
        }

        // Stochastic remainder selections
        let mut remaining_spots = self.num_parents - guaranteed_count;
        while remaining_spots > 0 {
            let chosen_index = *remaining_indices
                .iter()
                .max_by(|&&a, &&b| residual_probabilities[a].partial_cmp(&residual_probabilities[b]).unwrap())
                .unwrap();

            if rng.gen::<f64>() < residual_probabilities[chosen_index].to_f64().unwrap() {
                selected.set_row(parent_index, &population.row(chosen_index));
                parent_index += 1;
                remaining_spots -= 1;
            }

            // Reset probability after selection
            residual_probabilities[chosen_index] = T::zero();
        }

        selected
    }
}
