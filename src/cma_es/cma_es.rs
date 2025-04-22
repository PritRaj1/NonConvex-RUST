use nalgebra::{DVector, DMatrix};
use rand_distr::{Normal, Distribution};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::config::CMAESConf;
use rayon::prelude::*;

pub struct CMAES<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: CMAESConf,
    pub opt_prob: OptProb<T, F, G>,
    pub x: DVector<T>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    pub population: DMatrix<T>,
    pub fitness: DVector<T>,
    pub constraints: DVector<bool>,
    
    // Strategy parameters (private)
    mean: DVector<T>,
    pc: DVector<T>,          // Evolution path for c_mat
    ps: DVector<T>,          // Evolution path for sigma
    c_mat: DMatrix<T>,       // Covariance matrix
    b_mat: DMatrix<T>,       // Eigenvectors of c_mat
    d_vec: DVector<T>,       // Eigenvalues of c_mat
    sigma: T,                // Step size
    weights: DVector<T>,     // Recombination weights
    generation: usize,       // Counts the number of generations
    
    // Derived values (private)
    mu: usize,              // Number of parents
    lambda: usize,          // Population size
    mueff: T,               // Variance effective selection mass
    cc: T,                  // Time constant for cumulation for c_mat
    cs: T,                  // Time constant for cumulation for sigma
    c1: T,                  // Learning rate for rank-one update
    cmu: T,                 // Learning rate for rank-mu update
    damps: T,               // Damping for sigma
    chi_n: T,               // Expected norm of N(0,I)
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> CMAES<T, F, G> {
    pub fn new(conf: CMAESConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let n = init_x.len();
        let lambda = conf.population_size;
        let mu = conf.num_parents;
        let initial_sigma = conf.initial_sigma; // Get this before moving conf
        
        // Initialize weights
        let mut weights = DVector::zeros(mu);
        for i in 0..mu {
            weights[i] = T::ln(T::from_f64((lambda as f64 + 1.0) / 2.0).unwrap()) - 
                        T::ln(T::from_f64((i + 1) as f64).unwrap());
        }
        weights /= weights.sum();

        // Get initial fitness before moving opt_prob
        let initial_fitness = opt_prob.objective.f(&init_x);
        
        // Compute derived strategy parameters
        let mueff = T::one() / weights.map(|w| w * w).sum();
        let n_f = T::from_f64(n as f64).unwrap();
        
        let cc = (T::from_f64(4.0).unwrap() + mueff/n_f) / 
                (n_f + T::from_f64(4.0).unwrap() + T::from_f64(2.0).unwrap() * mueff/n_f);
        
        let cs = (mueff + T::from_f64(2.0).unwrap()) / 
                (n_f + mueff + T::from_f64(5.0).unwrap());
        
        let c1 = T::from_f64(2.0).unwrap() / 
                ((n_f + T::from_f64(1.3).unwrap()).powi(2) + mueff);
        
        let cmu = T::min(
            T::one() - c1,
            T::from_f64(2.0).unwrap() * 
            (mueff - T::from_f64(2.0).unwrap() + T::one()/mueff) / 
            ((n_f + T::from_f64(2.0).unwrap()).powi(2) + mueff)
        );
        
        let damps = T::one() + 
                   T::from_f64(2.0).unwrap() * 
                   T::max(T::zero(), 
                         T::sqrt((mueff - T::one())/(n_f + T::one())) - T::one()) + 
                   cs;
        
        let chi_n = T::sqrt(n_f) * 
                   (T::one() - T::one()/(T::from_f64(4.0).unwrap() * n_f) + 
                    T::one()/(T::from_f64(21.0).unwrap() * n_f.powi(2)));
        
        Self {
            conf: conf.clone(),
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            best_x: init_x.clone(),
            best_fitness: initial_fitness,
            population: DMatrix::zeros(lambda, n),
            fitness: DVector::zeros(lambda),
            constraints: DVector::from_element(lambda, true),
            mean: init_x,
            pc: DVector::zeros(n),
            ps: DVector::zeros(n),
            c_mat: DMatrix::identity(n, n),
            b_mat: DMatrix::identity(n, n),
            d_vec: DVector::from_element(n, T::one()),
            sigma: T::from_f64(initial_sigma).unwrap(),
            generation: 0,
            weights,
            mu,
            lambda,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
        }
    }

    pub fn step(&mut self) {
        let n = self.mean.len();
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();

        // Avoid looped stochasticity
        let random_numbers: Vec<Vec<f64>> = (0..self.lambda)
            .map(|_| (0..n).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        let results: Vec<_> = random_numbers.par_iter()
            .map(|z_vec| {
                let z = DVector::from_iterator(n, z_vec.iter().map(|&z| T::from_f64(z).unwrap()));
                let y = &self.b_mat * &self.d_vec.component_mul(&z);
                let mut x = self.mean.clone();
                for i in 0..n {
                    x[i] = x[i] + self.sigma * y[i];
                }
                
                let fitness = self.opt_prob.objective.f(&x);
                let constraint = self.opt_prob.is_feasible(&x);
                
                (x, fitness, constraint)
            })
            .collect();

        for (i, (x, fitness, constraint)) in results.iter().enumerate() {
            self.population.set_row(i, &x.transpose());
            self.fitness[i] = *fitness;
            self.constraints[i] = *constraint;
        }

        // Sort indices
        let mut indices: Vec<usize> = (0..self.lambda).collect();
        indices.sort_by(|&i, &j| {
            let feasible_i = self.constraints[i];
            let feasible_j = self.constraints[j];
            match (feasible_i, feasible_j) {
                (true, true) => self.fitness[j].partial_cmp(&self.fitness[i]).unwrap(),
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => self.fitness[j].partial_cmp(&self.fitness[i]).unwrap(),
            }
        });

        // Update mean
        let old_mean = self.mean.clone();
        self.mean = DVector::zeros(n);
        for i in 0..self.mu {
            let row = self.population.row(indices[i]).transpose();
            for j in 0..n {
                self.mean[j] += self.weights[i] * row[j];
            }
        }

        // Update evolution paths
        let mut y = DVector::zeros(n);
        for i in 0..n {
            y[i] = (self.mean[i] - old_mean[i]) / self.sigma;
        }
        
        // Compute conjugate evolution path
        let bdinvy = &self.b_mat * &self.d_vec.map(|d| T::one()/d).component_mul(&(&self.b_mat.transpose() * &y));
        let cs_factor = T::sqrt(self.cs * (T::from_f64(2.0).unwrap() - self.cs) * self.mueff);
        
        let mut ps_new = DVector::zeros(n);
        for i in 0..n {
            ps_new[i] = (T::one() - self.cs) * self.ps[i] + cs_factor * bdinvy[i];
        }
        self.ps = ps_new;

        // Update hsig
        let decay = T::one() - self.cs;
        let decay_pow = decay.powi(2 * self.generation as i32);
        let ps_norm = self.ps.dot(&self.ps).sqrt();
        let hsig = ps_norm / (T::sqrt(T::one() - decay_pow) * self.chi_n) < T::from_f64(1.4).unwrap();

        // Update pc
        let cc_factor = T::sqrt(self.cc * (T::from_f64(2.0).unwrap() - self.cc) * self.mueff);
        let hsig_t = if hsig { T::one() } else { T::zero() };
        
        let mut pc_new = DVector::zeros(n);
        for i in 0..n {
            pc_new[i] = (T::one() - self.cc) * self.pc[i] + hsig_t * cc_factor * y[i];
        }
        self.pc = pc_new;

        // Update covariance matrix
        let mut c_mat_new = DMatrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                c_mat_new[(i, j)] = (T::one() - self.c1 - self.cmu) * self.c_mat[(i, j)];
            }
        }
        
        // Rank-one update
        for i in 0..n {
            for j in 0..n {
                c_mat_new[(i, j)] += self.c1 * self.pc[i] * self.pc[j];
            }
        }
        
        // Rank-mu update
        for k in 0..self.mu {
            let mut y_k = DVector::zeros(n);
            for i in 0..n {
                y_k[i] = (self.population[(indices[k], i)] - old_mean[i]) / self.sigma;
            }
            
            for i in 0..n {
                for j in 0..n {
                    c_mat_new[(i, j)] += self.cmu * self.weights[k] * y_k[i] * y_k[j];
                }
            }
        }
        
        // Note: This is a simplified approximation using Cholesky-like decomposition
        let mut b_mat = DMatrix::identity(n, n);
        let mut d_vec = DVector::zeros(n);

        for i in 0..n {
            d_vec[i] = T::sqrt(c_mat_new[(i,i)].abs());
        }

        for i in 0..n {
            for j in (i+1)..n {
                if d_vec[i] > T::zero() && d_vec[j] > T::zero() {
                    b_mat[(i,j)] = c_mat_new[(i,j)] / (d_vec[i] * d_vec[j]);
                    b_mat[(j,i)] = b_mat[(i,j)];
                }
            }
        }

        self.c_mat = c_mat_new;
        self.b_mat = b_mat;
        self.d_vec = d_vec;
        
        // Step size update
        let ps_norm = self.ps.dot(&self.ps).sqrt();
        self.sigma *= T::exp(T::min(
            T::one(),
            (ps_norm/self.chi_n - T::one()) * self.cs/self.damps
        ));
        
        if self.constraints[indices[0]] && self.fitness[indices[0]] > self.best_fitness {
            self.best_fitness = self.fitness[indices[0]];
            self.best_x = self.population.row(indices[0]).transpose();
        }
        
        self.x = self.mean.clone();
        self.generation += 1;
    }
} 