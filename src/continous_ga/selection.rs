use nalgebra::{DVector, DMatrix};
use rand::Rng;
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub enum SelectionOperator {
    Proportional,
    Tournament,
    SRS,
}

pub struct Proportional{
    pub population_size: usize,
    pub num_parents: usize,
}

impl Proportional {
    pub fn new(population_size: usize, num_parents: usize) -> Self {
        Proportional { population_size, num_parents }
    }

    pub fn select<T: FloatNum>(&self, population: &DMatrix<T>, fitness: &DVector<T>, constraint: &DVector<bool>) -> DMatrix<T> {
        let sum = fitness.iter().fold(T::zero(), |acc, &x| acc + x);
        let llhoods = fitness.map(|x| x / sum);
        
        let mut selected = DMatrix::<T>::zeros(self.num_parents, population.ncols());
        let mut rng = rand::rng();

        for i in 0..self.num_parents {
            let mut r = T::from_f64(rng.random_range(0.0..1.0));
            for j in 0..population.nrows() {
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
