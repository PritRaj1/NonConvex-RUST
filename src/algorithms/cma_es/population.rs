use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

pub fn evaluate_samples<T: FloatNum>(
    samples: &[DVector<T>],
    mean: &DVector<T>,
    _C: &DMatrix<T>,
    opt_prob: &OptProb<T>,
    sigma: T,
) -> Vec<(DVector<T>, T, bool)> {
    samples.par_iter()
        .map(|x| {
            let mut sample = mean.clone();
            for i in 0..sample.len() {
                sample[i] = sample[i] + sigma * x[i];
            }
            
            let fitness = opt_prob.evaluate(&sample);
            let constraint = opt_prob.is_feasible(&sample);
            (sample, fitness, constraint)
        })
        .collect()
}

pub fn update_arrays<T: FloatNum>(
    population: &mut DMatrix<T>,
    fitness: &mut DVector<T>,
    constraints: &mut DVector<bool>,
    results: &[(DVector<T>, T, bool)]
) {
    for (i, (x, f, c)) in results.iter().enumerate() {
        population.set_row(i, &x.transpose());
        fitness[i] = *f;
        constraints[i] = *c;
    }
}

pub fn sort<T: FloatNum>(
    fitness: &DVector<T>,
    constraints: &DVector<bool>,
    lambda: usize
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..lambda).collect();
    indices.sort_by(|&i, &j| {
        let feasible_i = constraints[i];
        let feasible_j = constraints[j];
        match (feasible_i, feasible_j) {
            (true, true) => fitness[j].partial_cmp(&fitness[i]).unwrap(),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => fitness[j].partial_cmp(&fitness[i]).unwrap(),
        }
    });
    indices
} 