use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::FloatNumber as FloatNum;

pub fn compute_y<T: FloatNum>(mean: &DVector<T>, old_mean: &DVector<T>, sigma: T) -> DVector<T> {
    let mut y = DVector::zeros(mean.len());
    for i in 0..mean.len() {
        y[i] = (mean[i] - old_mean[i]) / sigma;
    }
    y
}

pub fn update_paths<T: FloatNum>(
    ps: &mut DVector<T>,
    b_mat: &DMatrix<T>,
    d_vec: &DVector<T>,
    cs: T,
    mueff: T,
    generation: usize,
    chi_n: T,
    y: &DVector<T>,
    n: usize
) -> bool {
    // Avoid double reference by dereferencing b_mat
    let b_mat = &*b_mat;
    let d_inv = d_vec.map(|d| T::one()/d);
    let b_trans_y = b_mat.transpose() * y;
    let bdinvy = b_mat * &d_inv.component_mul(&b_trans_y);
    
    let cs_factor = T::sqrt(cs * (T::from_f64(2.0).unwrap() - cs) * mueff);
    
    // Update ps
    let mut ps_new = DVector::zeros(n);
    for i in 0..n {
        ps_new[i] = (T::one() - cs) * ps[i] + cs_factor * bdinvy[i];
    }
    *ps = ps_new;

    // Update hsig
    let decay = T::one() - cs;
    let decay_pow = decay.powi(2 * generation as i32);
    let ps_norm = ps.dot(&ps).sqrt();
    ps_norm / (T::sqrt(T::one() - decay_pow) * chi_n) < T::from_f64(1.4).unwrap()
}

pub fn update_covariance<T: FloatNum>(
    c_mat: &mut DMatrix<T>,
    b_mat: &mut DMatrix<T>,
    d_vec: &mut DVector<T>,
    pc: &mut DVector<T>,
    y: &DVector<T>,
    hsig: bool,
    indices: &[usize],
    old_mean: &DVector<T>,
    c1: T,
    cmu: T,
    cc: T,
    mueff: T,
    population: &DMatrix<T>,
    weights: &DVector<T>,
    sigma: T,
    num_parents: usize,
    n: usize
) {
    let mut c_mat_new = DMatrix::zeros(n, n);
    let factor = T::one() - c1 - cmu;
    
    // Base update
    for i in 0..n {
        for j in 0..n {
            c_mat_new[(i, j)] = factor * c_mat[(i, j)];
        }
    }
    
    // Update pc
    let cc_factor = T::sqrt(cc * (T::from_f64(2.0).unwrap() - cc) * mueff);
    let hsig_t = if hsig { T::one() } else { T::zero() };
    
    // Update pc with single loop
    for i in 0..n {
        pc[i] = (T::one() - cc) * pc[i] + hsig_t * cc_factor * y[i];
    }

    // Rank-one update with single loop over upper triangle
    for i in 0..n {
        for j in i..n {
            let val = c1 * pc[i] * pc[j];
            c_mat_new[(i, j)] += val;
            if i != j {
                c_mat_new[(j, i)] += val;
            }
        }
    }

    // Rank-mu update
    for k in 0..num_parents {
        if k >= indices.len() || k >= weights.len() {
            continue;
        }
        let idx = indices[k];
        if idx >= population.nrows() {
            continue;
        }
        let w = weights[k];
        
        let mut y_k = DVector::zeros(n);
        for i in 0..n {
            if i < population.ncols() {
                y_k[i] = (population[(idx, i)] - old_mean[i]) / sigma;
            }
        }
        
        // Update c_mat_new with upper triangle only
        for i in 0..n {
            for j in i..n {
                let val = cmu * w * y_k[i] * y_k[j];
                c_mat_new[(i, j)] += val;
                if i != j {
                    c_mat_new[(j, i)] += val;
                }
            }
        }
    }

    // Update matrices
    *c_mat = c_mat_new;

    // Symmetric power iteration for eigendecomposition
    for i in 0..n {
        // Initialize random vector
        let mut v = DVector::from_fn(n, |_, _| {
            T::from_f64(rand::random::<f64>()).unwrap() * T::from_f64(2.0).unwrap() - T::one()
        });
        
        // Manual norm calculation and normalization
        let v_norm = T::sqrt(v.dot(&v));
        for j in 0..n {
            v[j] = v[j] / v_norm;
        }

        // Power iteration
        for _ in 0..20 {  // Usually converges in < 20 iterations
            let v_new = &*c_mat * &v;  // Proper borrowing
            let norm = T::sqrt(v_new.dot(&v_new));
            
            if norm > T::from_f64(1e-10).unwrap() {
                for j in 0..n {
                    v[j] = v_new[j] / norm;
                }
            }
        }

        // Extract eigenvalue and update matrices
        let c_v = &*c_mat * &v;
        let eigenvalue = v.dot(&c_v);  // v^T * C * v
        d_vec[i] = T::sqrt(T::max(eigenvalue.abs(), T::from_f64(1e-20).unwrap()));
        b_mat.set_column(i, &v);

        // Deflate matrix to find next eigenpair
        let vvt = &v * &v.transpose();  // Outer product
        for j in 0..n {
            for k in 0..n {
                c_mat[(j,k)] -= eigenvalue * vvt[(j,k)];
            }
        }
    }

    // Restore C = BDB^T
    let d_mat = DMatrix::from_diagonal(&d_vec.map(|x| x * x));
    let temp = &*b_mat * &d_mat;
    *c_mat = &temp * &b_mat.transpose();
} 