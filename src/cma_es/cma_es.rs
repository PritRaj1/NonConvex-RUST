use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::config::CMAESConf;
use crate::cma_es::parameters::Parameters;
use crate::cma_es::population::{evaluate_samples, update_arrays, sort};
use crate::cma_es::evolution::{update_paths, update_covariance, compute_y};
use rand_distr::{Normal, Distribution};

pub struct CMAES<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: CMAESConf,
    pub opt_prob: OptProb<T, F, G>,
    pub x: DVector<T>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    pub population: DMatrix<T>,
    pub fitness: DVector<T>,
    pub constraints: DVector<bool>,
    
    // Strategy parameters
    pub mean: DVector<T>,
    pub pc: DVector<T>,          // Evolution path for c_mat
    pub ps: DVector<T>,          // Evolution path for sigma
    pub c_mat: DMatrix<T>,       // Covariance matrix
    pub b_mat: DMatrix<T>,       // Eigenvectors of c_mat
    pub d_vec: DVector<T>,       // Eigenvalues of c_mat
    pub sigma: T,                // Step size
    pub weights: DVector<T>,     // Recombination weights
    pub generation: usize,       // Counts the number of generations
    
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

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> CMAES<T, F, G> {
    pub fn new(conf: CMAESConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let n = init_x.len();
        let params = Parameters::new(&conf, &init_x);
        let initial_fitness = opt_prob.objective.f(&init_x);

        // Initial population
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(params.lambda);
        
        for _ in 0..params.lambda {
            let mut z = DVector::zeros(n);
            for j in 0..n {
                z[j] = T::from_f64(normal.sample(&mut rng)).unwrap();
            }
            samples.push(z);
        }
        let results = evaluate_samples(
            &samples, &init_x, &DMatrix::identity(n, n), 
            &DVector::from_element(n, T::one()),
            T::from_f64(conf.initial_sigma).unwrap(),
            &opt_prob, n
        );

        let mut population = DMatrix::zeros(params.lambda, n);
        let mut fitness = DVector::zeros(params.lambda);
        let mut constraints = DVector::from_element(params.lambda, false);

        for (i, (x, f, c)) in results.iter().enumerate() {
            population.set_row(i, &x.transpose());
            fitness[i] = *f;
            constraints[i] = *c;
        }

        Self {
            conf: conf.clone(),
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            best_x: init_x.clone(),
            best_fitness: initial_fitness,
            population,
            fitness,
            constraints,
            mean: init_x,
            pc: DVector::zeros(n),
            ps: DVector::zeros(n),
            c_mat: DMatrix::identity(n, n),
            b_mat: DMatrix::identity(n, n),
            d_vec: DVector::from_element(n, T::one()),
            sigma: T::from_f64(conf.initial_sigma).unwrap(),
            generation: 0,
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

    fn generate_samples(&self, n: usize) -> Vec<DVector<T>> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        
        (0..self.lambda)
            .map(|_| {
                DVector::from_iterator(n, 
                    (0..n).map(|_| T::from_f64(normal.sample(&mut rng)).unwrap())
                )
            })
            .collect()
    }

    pub fn step(&mut self) {
        let n = self.mean.len();
        
        // Population updates
        let samples = self.generate_samples(n);
        let results = evaluate_samples(
            &samples, &self.mean, &self.b_mat, &self.d_vec,
            self.sigma, &self.opt_prob, n
        );
        update_arrays(&mut self.population, &mut self.fitness, &mut self.constraints, &results);
        
        // Sort and update mean - weigthed average of μ selected points from sample space
        let indices = sort(&self.fitness, &self.constraints, self.lambda);
        let old_mean = self.mean.clone();
        self.mean = DVector::zeros(n);
        for i in 0..self.mu {
            let row = self.population.row(indices[i]).transpose();
            for j in 0..n {
                self.mean[j] += self.weights[i] * row[j]; 
            }
        }

        // Evolution path updates - update step size and covariance matrix
        let y = compute_y(&self.mean, &old_mean, self.sigma);
        let hsig = update_paths(
            &mut self.ps, &self.b_mat, &self.d_vec,
            self.cs, self.mueff, self.generation, self.chi_n,
            &y, n
        );
        update_covariance(
            &mut self.c_mat, &mut self.b_mat, &mut self.d_vec,
            &mut self.pc, &y, hsig, &indices, &old_mean,
            self.c1, self.cmu, self.cc, self.mueff,
            &self.population, &self.weights, self.sigma,
            self.mu,
            n
        );
        
        let ps_norm = self.ps.dot(&self.ps).sqrt();
        self.sigma *= T::exp(T::min(
            T::one(),
            (ps_norm/self.chi_n - T::one()) * self.cs/self.damps
        ));

        // Update best solution
        if self.constraints[indices[0]] && self.fitness[indices[0]] > self.best_fitness {
            self.best_fitness = self.fitness[indices[0]];
            self.best_x = self.population.row(indices[0]).transpose();
        }
        
        self.x = self.mean.clone();
        self.generation += 1;
    }
} 