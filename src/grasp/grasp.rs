use nalgebra::DVector;
use rand::Rng;
use rayon::prelude::*;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::config::GRASPConf;

pub struct GRASP<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: GRASPConf,
    pub x: DVector<T>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> GRASP<T, F, G> {
    pub fn new(conf: GRASPConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let fitness = opt_prob.objective.f(&init_x);
        
        Self {
            conf,
            x: init_x.clone(),
            opt_prob,
            best_x: init_x,
            best_fitness: fitness,
        }
    }

    // Greedy randomized construction phase
    pub fn construct_solution(&self) -> DVector<T> {
        let candidates: Vec<DVector<T>> = (0..self.conf.num_candidates)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng(); // Create new RNG for each thread
                let mut candidate = DVector::zeros(self.x.len());
                for i in 0..self.x.len() {
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
                let fa = self.opt_prob.objective.f(a);
                let fb = self.opt_prob.objective.f(b);
                fa.partial_cmp(&fb).unwrap()
            })
            .unwrap_or(self.x.clone())
    }

    // Local search phase
    pub fn local_search(&self, solution: &DVector<T>) -> DVector<T> {
        let mut current = solution.clone();
        let mut current_fitness = self.opt_prob.objective.f(&current);
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
                    let fa = self.opt_prob.objective.f(a);
                    let fb = self.opt_prob.objective.f(b);
                    fa.partial_cmp(&fb).unwrap()
                }) 
            {
                let neighbor_fitness = self.opt_prob.objective.f(&best_neighbor);
                if neighbor_fitness > current_fitness {
                    current = best_neighbor;
                    current_fitness = neighbor_fitness;
                    improved = true;
                }
            }
        }

        current
    }

    pub fn step(&mut self) {
        let solution = self.construct_solution();
        let improved_solution = self.local_search(&solution);
        
        let fitness = self.opt_prob.objective.f(&improved_solution);
        if fitness > self.best_fitness && self.opt_prob.is_feasible(&improved_solution) {
            self.best_fitness = fitness;
            self.best_x = improved_solution.clone();
        }
        
        self.x = improved_solution;
    }
} 