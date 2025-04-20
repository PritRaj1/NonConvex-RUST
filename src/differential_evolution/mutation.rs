use nalgebra::{DVector, DMatrix};
use rand::Rng;
use crate::utils::opt_prob::FloatNumber as FloatNum;

fn get_random_indices(count: usize, exclude: usize, pop_size: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    let mut indices = Vec::new();
    while indices.len() < count {
        let idx = rng.random_range(0..pop_size);
        if idx != exclude && !indices.contains(&idx) {
            indices.push(idx);
        }
    }
    indices
}

pub fn rand1_bin<T: FloatNum>(
    population: &DMatrix<T>,
    target_idx: usize,
    f: T,
    cr: T,
) -> DVector<T> {
    let mut rng = rand::rng();
    let pop_size = population.nrows();
    let dim = population.ncols();
    
    let indices = get_random_indices(3, target_idx, pop_size);
    
    let x_r1 = population.row(indices[0]).transpose();
    let x_r2 = population.row(indices[1]).transpose();
    let x_r3 = population.row(indices[2]).transpose();
    let donor = x_r1 + (x_r2 - x_r3) * f;
    
    let mut trial = population.row(target_idx).transpose();
    let j_rand = rng.random_range(0..dim);

    for j in 0..dim {
        if rng.random::<f64>() < cr.to_f64().unwrap() || j == j_rand {
            trial[j] = donor[j];
        }
    }

    trial
}

pub fn best1_bin<T: FloatNum>(
    population: &DMatrix<T>,
    best_x: &DVector<T>,
    target_idx: usize,
    f: T,
    cr: T,
) -> DVector<T> {
    let mut rng = rand::rng();
    let pop_size = population.nrows();
    let dim = population.ncols();
    
    let indices = get_random_indices(2, target_idx, pop_size);
    
    let x_r1 = population.row(indices[0]).transpose();
    let x_r2 = population.row(indices[1]).transpose();
    let donor = best_x + (x_r1 - x_r2) * f;
    
    let mut trial = population.row(target_idx).transpose();
    let j_rand = rng.random_range(0..dim);

    for j in 0..dim {
        if rng.random::<f64>() < cr.to_f64().unwrap() || j == j_rand {
            trial[j] = donor[j];
        }
    }

    trial
}

pub fn rand_to_best1_bin<T: FloatNum>(
    population: &DMatrix<T>,
    best_x: &DVector<T>,
    target_idx: usize,
    f: T,
    cr: T,
) -> DVector<T> {
    let mut rng = rand::rng();
    let pop_size = population.nrows();
    let dim = population.ncols();
    
    let indices = get_random_indices(2, target_idx, pop_size);
    let x_r1 = population.row(indices[0]);
    let x_r2 = population.row(indices[1]);
    let x_i = population.row(target_idx);
    
    let mut trial = x_i.transpose();
    let j_rand = rng.random_range(0..dim);

    for j in 0..dim {
        if rng.random::<f64>() < cr.to_f64().unwrap() || j == j_rand {
            trial[j] = x_i[j] + f * (best_x[j] - x_i[j]) + f * (x_r1[j] - x_r2[j]);
        }
    }

    trial
}

pub fn best2_bin<T: FloatNum>(
    population: &DMatrix<T>,
    best_x: &DVector<T>,
    target_idx: usize,
    f: T,
    cr: T,
) -> DVector<T> {
    let mut rng = rand::rng();
    let pop_size = population.nrows();
    let dim = population.ncols();
    
    let indices = get_random_indices(4, target_idx, pop_size);
    let x_r1 = population.row(indices[0]);
    let x_r2 = population.row(indices[1]);
    let x_r3 = population.row(indices[2]);
    let x_r4 = population.row(indices[3]);
    
    let mut trial = population.row(target_idx).transpose();
    let j_rand = rng.random_range(0..dim);

    for j in 0..dim {
        if rng.random::<f64>() < cr.to_f64().unwrap() || j == j_rand {
            trial[j] = best_x[j] + f * (x_r1[j] + x_r2[j] - x_r3[j] - x_r4[j]);
        }
    }

    trial
}

pub fn rand2_bin<T: FloatNum>(
    population: &DMatrix<T>,
    target_idx: usize,
    f: T,
    cr: T,
) -> DVector<T> {
    let mut rng = rand::rng();
    let pop_size = population.nrows();
    let dim = population.ncols();
    
    let indices = get_random_indices(5, target_idx, pop_size);
    let x_r1 = population.row(indices[0]);
    let x_r2 = population.row(indices[1]);
    let x_r3 = population.row(indices[2]);
    let x_r4 = population.row(indices[3]);
    let x_r5 = population.row(indices[4]);
    
    let mut trial = population.row(target_idx).transpose();
    let j_rand = rng.random_range(0..dim);

    for j in 0..dim {
        if rng.random::<f64>() < cr.to_f64().unwrap() || j == j_rand {
            trial[j] = x_r1[j] + f * (x_r2[j] + x_r3[j] - x_r4[j] - x_r5[j]);
        }
    }

    trial
}

