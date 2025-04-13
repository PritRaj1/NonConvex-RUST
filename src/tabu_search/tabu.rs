use nalgebra::DVector;
use rand::Rng;
use std::collections::VecDeque; // Double-ended queue
use rayon::prelude::*;
use crate::utils::config::TabuConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};

pub struct TabuSearch<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: TabuConf,
    pub x: DVector<T>,
    pub tabu_list: VecDeque<DVector<T>>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> TabuSearch<T, F, G> {
    pub fn new(conf: TabuConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let fitness = opt_prob.objective.f(&init_x);
        let mut tabu_list = VecDeque::with_capacity(conf.tabu_list_size);
        tabu_list.push_back(init_x.clone());
        
        Self {
            conf,
            x: init_x.clone(),
            tabu_list,
            opt_prob,
            best_x: init_x,
            best_fitness: fitness,
        }
    }

    fn generate_neighbors(&self, center: &DVector<T>) -> Vec<DVector<T>> {
        (0..self.conf.num_neighbors)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                let mut neighbor = center.clone();
                
                // Randomly perturb some dimensions
                for i in 0..center.len() {
                    if rng.random_bool(self.conf.perturbation_prob) {
                        let perturbation = T::from_f64(
                            rng.random_range(-self.conf.step_size..self.conf.step_size)
                        ).unwrap();
                        neighbor[i] += perturbation;
                    }
                }
                
                neighbor
            })
            .collect()
    }

    // Check if a neighbor is tabu
    fn is_tabu(&self, x: &DVector<T>) -> bool {
        self.tabu_list.iter().any(|tabu_x| {
            let diff = x - tabu_x;
            diff.iter().all(|&d| d.abs() < T::from_f64(self.conf.tabu_threshold).unwrap())
        })
    }

    pub fn step(&mut self) {
        let neighbors = self.generate_neighbors(&self.x);
        
        let evaluations: Vec<(DVector<T>, T, bool)> = neighbors
            .into_par_iter()
            .filter_map(|neighbor| {
                let is_feasible = match &self.opt_prob.constraints {
                    Some(constraints) => constraints.g(&neighbor),
                    None => true,
                };
                
                let fitness = self.opt_prob.objective.f(&neighbor);
                if is_feasible && (!self.is_tabu(&neighbor) || fitness > self.best_fitness) {
                    Some((neighbor, fitness, is_feasible))
                } else {
                    None
                }
            })
            .collect();

        // Find best (maximum) non-tabu neighbor
        if let Some((best_neighbor, best_neighbor_fitness, _)) = evaluations
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            self.x = best_neighbor.clone();
            
            // Update fitness and tabu list
            if best_neighbor_fitness > self.best_fitness {
                self.best_fitness = best_neighbor_fitness;
                self.best_x = best_neighbor.clone();
            }
            
            self.tabu_list.push_back(best_neighbor);
            if self.tabu_list.len() > self.conf.tabu_list_size {
                self.tabu_list.pop_front();
            }
        }
    }
} 