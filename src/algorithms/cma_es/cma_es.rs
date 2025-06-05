use rand;
use rand_distr::{Normal, Distribution};
use nalgebra::{
    allocator::Allocator, 
    DefaultAllocator, 
    Dim, 
    OMatrix, 
    OVector,
    U1,
    Dyn,
};

use crate::utils::config::CMAESConf;
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

use crate::algorithms::cma_es::{
    parameters::Parameters,
    population::{evaluate_samples, update_arrays, sort},
    evolution::{update_paths, update_covariance, compute_y},
};

pub struct CMAES<T: FloatNum + Send + Sync, N: Dim, D: Dim>
where 
    DefaultAllocator: Allocator<D> 
                     + Allocator<N, D>
                     + Allocator<N>
                     + Allocator<D, D>
{
    pub conf: CMAESConf,
    pub opt_prob: OptProb<T, D>,
    pub st: State<T, N, D>,
    
    // Strategy parameters
    pub mean: OVector<T, D>,
    pub pc: OVector<T, D>,          // Evolution path for c_mat
    pub ps: OVector<T, D>,          // Evolution path for sigma
    pub c_mat: OMatrix<T, D, D>,       // Covariance matrix
    pub b_mat: OMatrix<T, D, D>,       // Eigenvectors of c_mat
    pub d_vec: OVector<T, D>,       // Eigenvalues of c_mat
    pub sigma: T,                // Step size
    pub weights: OVector<T, Dyn>,     // Recombination weights
    
    // Derived values
    pub mu: usize,              // Number of parents < λ
    pub lambda: usize,          // Population size
    pub mueff: T,               // Variance effective selection mass
    pub cc: T,                  // Time constant for cumulation for c_mat
    pub cs: T,                  // Time constant for cumulation for sigma
    pub c1: T,                  // Learning rate for rank-one update
    pub cmu: T,                 // Learning rate for rank-mu update
    pub damps: T,               // Damping for sigma
    pub chi_n: T,               // Expected norm of N(0,I)
}

impl<T: FloatNum, N: Dim, D: Dim> CMAES<T, N, D> 
where 
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> 
                    + Allocator<N, D>
                    + Allocator<N>
                    + Allocator<D, D>
                    + Allocator<U1, D>
{
    pub fn new(conf: CMAESConf, init_pop: OMatrix<T, U1, D>, opt_prob: OptProb<T, D>) -> Self {
        let init_x: OVector<T, D> = init_pop.row(0).transpose().into_owned();

        let n = init_x.len();
        let params = Parameters::new(&conf, &init_x);
        
        // Initial population
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(params.lambda);
        
        for _ in 0..params.lambda {
            let mut z = init_x.clone() * T::from_f64(0.0).unwrap();
            for j in 0..n {
                z[j] = T::from_f64(normal.sample(&mut rng)).unwrap();
            }
            samples.push(z);
        }

        let b_mat = OMatrix::<T, D, D>::identity_generic(D::from_usize(n), D::from_usize(n));
        let d_vec: OVector<T, D> = OVector::from_element_generic(D::from_usize(n), U1, T::one());

        let results = evaluate_samples(
            &samples,
            &init_x,
            &b_mat.clone(),                    
            &d_vec.clone(),                    
            &opt_prob,
            T::from_f64(conf.initial_sigma).unwrap(),
        );

        let mut population: OMatrix<T, N, D> = OMatrix::from_element_generic(N::from_usize(params.lambda), D::from_usize(n), T::zero());
        let mut fitness: OVector<T, N> = OVector::from_element_generic(N::from_usize(params.lambda), U1, T::zero());
        let mut constraints: OVector<bool, N> = OVector::from_element_generic(N::from_usize(params.lambda), U1, true);

        for (i, (x, f, c)) in results.iter().enumerate() {
            population.row_mut(i).copy_from(&x.transpose());
            fitness[i] = *f;
            constraints[i] = *c;
        }

        let best_f = fitness[0];
        let best_x = population.row(0).transpose();

        let st = State {
            best_x: best_x.clone(),
            best_f,
            pop: population.clone(),
            fitness: fitness.clone(),
            constraints: constraints.clone(),
            iter: 0,
        };

        Self {
            conf: conf.clone(),
            opt_prob,
            st,
            mean: init_x,
            pc: OVector::zeros_generic(D::from_usize(n), U1),
            ps: OVector::zeros_generic(D::from_usize(n), U1),
            c_mat: OMatrix::<T, D, D>::identity_generic(D::from_usize(n), D::from_usize(n)),
            b_mat: OMatrix::<T, D, D>::identity_generic(D::from_usize(n), D::from_usize(n)),
            d_vec: OVector::from_element_generic(D::from_usize(n), U1, T::one()),
            sigma: T::from_f64(conf.initial_sigma).unwrap(),
            weights: params.weights,
            mu: params.mu,
            lambda: params.lambda,
            mueff: params.mueff,
            cc: params.cc,
            cs: params.cs,
            c1: params.c1,
            cmu: params.cmu,
            damps: params.damps,
            chi_n: params.chi_n,
        }
    }

    fn generate_samples(&self, n: usize) -> Vec<OVector<T, D>> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        
        (0..self.lambda)
        .map(|_| {
            let iter = (0..n).map(|_| T::from_f64(normal.sample(&mut rng)).unwrap());
            OVector::<T, D>::from_iterator_generic(D::from_usize(n), U1, iter)
        })
        .collect()
    }
}

impl<T: FloatNum, N: Dim, D: Dim> OptimizationAlgorithm<T, N, D> for CMAES<T, N, D> 
where 
    OVector<T, D>: Send + Sync,
    OMatrix<T, D, D>: Send + Sync,
    DefaultAllocator: Allocator<D> 
                    + Allocator<N, D>
                    + Allocator<N>
                    + Allocator<D, D>
                    + Allocator<U1, D>
{
    fn step(&mut self) {
        let n = self.mean.len();
        
        // Generate and evaluate new samples
        let samples = self.generate_samples(n);
        let results = evaluate_samples(
            &samples, 
            &self.mean, 
            &self.b_mat,
            &self.d_vec,
            &self.opt_prob, 
            self.sigma
        );
        
        // Update population arrays
        update_arrays(&mut self.st.pop, &mut self.st.fitness, &mut self.st.constraints, &results);
        
        // Sort and update mean
        let indices = sort(&self.st.fitness, &self.st.constraints, self.lambda);
        let old_mean = self.mean.clone();
        self.mean = OVector::zeros_generic(D::from_usize(n), U1);
        
        for i in 0..self.mu {
            let row = self.st.pop.row(indices[i]).transpose();
            for j in 0..n {
                self.mean[j] += self.weights[i] * row[j];
            }
        }

        // Evolution path updates
        let y = compute_y(&self.mean, &old_mean, self.sigma);
        let hsig = update_paths(
            &mut self.ps,
            &self.b_mat,
            &self.d_vec,
            self.cs,
            self.mueff,
            self.st.iter,
            self.chi_n,
            &y,
            n
        );

        update_covariance(
            &mut self.c_mat,
            &mut self.b_mat,
            &mut self.d_vec,
            &mut self.pc,
            &y,
            hsig,
            &indices,
            &old_mean,
            self.c1,
            self.cmu,
            self.cc,
            self.mueff,
            &self.st.pop,
            &self.weights,
            self.sigma,
            self.mu,
            n
        );

        // Update step size
        let ps_norm = self.ps.dot(&self.ps).sqrt();
        self.sigma *= T::exp(T::min(
            T::one(),
            (ps_norm/self.chi_n - T::one()) * self.cs/self.damps
        ));

        // Update best solution if improved
        if self.st.constraints[indices[0]] && self.st.fitness[indices[0]] > self.st.best_f {
            self.st.best_f = self.st.fitness[indices[0]];
            self.st.best_x = self.st.pop.row(indices[0]).transpose();
        }

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
} 