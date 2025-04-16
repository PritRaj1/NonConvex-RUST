use nalgebra::DVector;
use rand::Rng;
use crate::utils::config::TabuConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::tabu_search::tabu_list::{TabuList, TabuType};
use rayon::prelude::*;
pub struct TabuSearch<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: TabuConf,
    pub x: DVector<T>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    tabu_list: TabuList<T>,
    iterations_since_improvement: usize,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> TabuSearch<T, F, G> {
    pub fn new(conf: TabuConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let fitness = opt_prob.objective.f(&init_x);
        let tabu_type = TabuType::from(&conf);
        
        Self {
            conf: conf.clone(),
            x: init_x.clone(),
            opt_prob,
            best_x: init_x,
            best_fitness: fitness,
            tabu_list: TabuList::new(conf.tabu_list_size, tabu_type),
            iterations_since_improvement: 0,
        }
    }

    fn generate_neighbor(&self, rng: &mut impl Rng) -> DVector<T> {
        let mut neighbor = self.x.clone();
        neighbor.iter_mut().for_each(|val| {
            if rng.random_bool(self.conf.perturbation_prob) {
                *val += T::from_f64(
                    rng.random_range(-self.conf.step_size..self.conf.step_size)
                ).unwrap();
            }
        });
        neighbor
    }

    fn evaluate_neighbor(&self, neighbor: &DVector<T>) -> Option<T> {
        if self.opt_prob.is_feasible(neighbor) 
            && !self.tabu_list.is_tabu(neighbor, T::from_f64(self.conf.tabu_threshold).unwrap()) {
            Some(self.opt_prob.objective.f(neighbor))
        } else {
            None
        }
    }

    pub fn step(&mut self) {
        let mut best_neighbor = self.x.clone();
        let mut best_neighbor_fitness = T::neg_infinity();
        
        // Generate and evaluate neighborhood
        let neighbors: Vec<_> = (0..self.conf.num_neighbors)
            .into_par_iter()
            .map(|_| {
                let mut local_rng = rand::rng();
                let neighbor = self.generate_neighbor(&mut local_rng);
                let fitness = self.evaluate_neighbor(&neighbor);
                (neighbor, fitness)
            })
            .filter_map(|(neighbor, fitness)| {
                fitness.map(|f| (neighbor, f))
            })
            .collect();

        for (neighbor, fitness) in neighbors {
            if fitness > best_neighbor_fitness {
                best_neighbor = neighbor;
                best_neighbor_fitness = fitness;
            }
        }
        
        // Update current solution and best solution if improved
        if best_neighbor_fitness > T::neg_infinity() {
            self.tabu_list.update(self.x.clone(), self.iterations_since_improvement);
            
            self.x = best_neighbor.clone();

            if best_neighbor_fitness > self.best_fitness {
                self.best_fitness = best_neighbor_fitness;
                self.best_x = best_neighbor;
                self.iterations_since_improvement = 0;
            } else {
                self.iterations_since_improvement += 1;
            } 
        }
    }
} 