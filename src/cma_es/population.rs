use nalgebra::{DVector, DMatrix};
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};

pub fn generate_samples(lambda: usize, n: usize) -> Vec<Vec<f64>> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();
    (0..lambda)
        .map(|_| (0..n).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

pub fn evaluate_samples<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>>(
    samples: &[Vec<f64>],
    mean: &DVector<T>,
    b_mat: &DMatrix<T>,
    d_vec: &DVector<T>,
    sigma: T,
    opt_prob: &OptProb<T, F, G>,
    n: usize
) -> Vec<(DVector<T>, T, bool)> {
    samples.par_iter()
        .map(|z_vec| {
            let z = DVector::from_iterator(n, z_vec.iter().map(|&z| T::from_f64(z).unwrap()));
            let y = b_mat * &d_vec.component_mul(&z);
            let mut x = mean.clone();
            for i in 0..n {
                x[i] = x[i] + sigma * y[i];
            }
            
            let fitness = opt_prob.objective.f(&x);
            let constraint = opt_prob.is_feasible(&x);
            
            (x, fitness, constraint)
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