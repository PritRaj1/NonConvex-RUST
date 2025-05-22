use nalgebra::{DVector, DMatrix};
use rand::Rng;
use rayon::prelude::*;

use crate::utils::config::GRASPConf;
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

pub struct GRASP<T: FloatNum> 
where 
    T: Send + Sync 
{
    pub conf: GRASPConf,
    pub st: State<T>,
    pub opt_prob: OptProb<T>,
}

impl<T: FloatNum> GRASP<T> 
where 
    T: Send + Sync 
{
    pub fn new(conf: GRASPConf, init_pop: DMatrix<T>, opt_prob: OptProb<T>) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        
        Self {
            conf,
            st: State {

                best_x: init_x.clone(),
                best_f: best_f,
                pop: DMatrix::from_columns(&[init_x.clone()]),
                fitness: vec![best_f].into(),
                constraints: vec![opt_prob.is_feasible(&init_x)].into(),
                iter: 1
            },
            opt_prob,
        }
    }

    // Greedy randomized construction phase
    pub fn construct_solution(&self) -> DVector<T> {
        let candidates: Vec<DVector<T>> = (0..self.conf.num_candidates)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng(); // Create new RNG for each thread
                let mut candidate = DVector::zeros(self.st.best_x.len());
                for i in 0..self.st.best_x.len() {
                    // Get bounds if they exist
                    let lb = self.opt_prob.objective.x_lower_bound(&candidate)
                        .map_or(T::from_f64(-10.0).unwrap(), |v| v[i]);
                    let ub = self.opt_prob.objective.x_upper_bound(&candidate)
                        .map_or(T::from_f64(10.0).unwrap(), |v| v[i]);
                    
                    // Generate value within restricted candidate list (RCL)
                    let alpha = T::from_f64(self.conf.alpha).unwrap();
                    let rcl_min = lb * (T::one() - alpha) + ub * alpha;
                    let rcl_max = lb * alpha + ub * (T::one() - alpha);
                    
                    candidate[i] = T::from_f64(
                        rng.random_range(rcl_min.to_f64().unwrap()..rcl_max.to_f64().unwrap())
                    ).unwrap();
                }
                candidate
            })
            .collect();

        // Select best feasible candidate
        candidates.into_iter()
            .filter(|c| self.opt_prob.is_feasible(c))
            .max_by(|a, b| {
                let fa = self.opt_prob.evaluate(a);
                let fb = self.opt_prob.evaluate(b);
                fa.partial_cmp(&fb).unwrap()
            })
            .unwrap_or(self.st.best_x.clone())
    }

    // Local search phase
    pub fn local_search(&self, solution: &DVector<T>) -> DVector<T> {
        let mut current = solution.clone();
        let mut current_fitness = self.opt_prob.evaluate(&current);
        let mut improved = true;

        while improved {
            improved = false;
            
            // Generate and evaluate neighborhood in parallel
            let neighbors: Vec<DVector<T>> = (0..self.conf.num_neighbors)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::rng();
                    let mut neighbor = current.clone();
                    
                    // Perturb random dimensions
                    for i in 0..neighbor.len() {
                        if rng.random_bool(self.conf.perturbation_prob) {
                            neighbor[i] += T::from_f64(
                                rng.random_range(-self.conf.step_size..self.conf.step_size)
                            ).unwrap();
                        }
                    }
                    neighbor
                })
                .collect();

            // Find best feasible neighbor
            if let Some(best_neighbor) = neighbors.into_iter()
                .filter(|n| self.opt_prob.is_feasible(n))
                .max_by(|a, b| {
                    let fa = self.opt_prob.evaluate(a);
                    let fb = self.opt_prob.evaluate(b);
                    fa.partial_cmp(&fb).unwrap()
                }) 
            {
                let neighbor_fitness = self.opt_prob.evaluate(&best_neighbor);
                if neighbor_fitness > current_fitness {
                    current = best_neighbor;
                    current_fitness = neighbor_fitness;
                    improved = true;
                }
            }
        }

        current
    }
}

impl<T: FloatNum> OptimizationAlgorithm<T> for GRASP<T>{
    fn step(&mut self) {
        let solution = self.construct_solution();
        let improved_solution = self.local_search(&solution);
        
        let fitness = self.opt_prob.evaluate(&improved_solution);
        if fitness > self.st.best_f && self.opt_prob.is_feasible(&improved_solution) {
            self.st.best_f = fitness;
            self.st.best_x = improved_solution.clone();
        }
        
        self.st.pop = DMatrix::from_columns(&[improved_solution.clone()]);
        self.st.fitness[0] = fitness;
        self.st.constraints[0] = self.opt_prob.is_feasible(&improved_solution);

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T> {
        &self.st
    }
} 