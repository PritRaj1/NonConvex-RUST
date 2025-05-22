use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use rand::Rng;

use crate::utils::config::TabuConf;
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

use crate::algorithms::tabu_search::tabu_list::{TabuList, TabuType};

pub struct TabuSearch<T: FloatNum> 
where 
    T: Send + Sync 
{
    pub conf: TabuConf,
    pub opt_prob: OptProb<T>,
    pub x: DVector<T>,
    pub st: State<T>,
    tabu_list: TabuList<T>,
    iterations_since_improvement: usize,
}

impl<T: FloatNum> TabuSearch<T> 
where 
    T: Send + Sync 
{
    pub fn new(conf: TabuConf, init_pop: DMatrix<T>, opt_prob: OptProb<T>) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let tabu_type = TabuType::from(&conf);
        
        Self {
            conf: conf.clone(),
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            st: State{
                best_x: init_x.clone().into(),
                best_f,
                pop: DMatrix::from_columns(&[init_x.clone()]),
                fitness: vec![best_f].into(),
                constraints: vec![opt_prob.is_feasible(&init_x)].into(),
                iter: 1,
            },
            tabu_list: TabuList::new(conf.common.tabu_list_size, tabu_type),
            iterations_since_improvement: 0,
        }
    }

    fn generate_neighbor(&self, rng: &mut impl Rng) -> DVector<T> {
        let mut neighbor = self.x.clone();
        neighbor.iter_mut().for_each(|val| {
            if rng.random_bool(self.conf.common.perturbation_prob) {
                *val += T::from_f64(
                    rng.random_range(-self.conf.common.step_size..self.conf.common.step_size)
                ).unwrap();
            }
        });
        neighbor
    }

    fn evaluate_neighbor(&self, neighbor: &DVector<T>) -> Option<T> {
        if self.opt_prob.is_feasible(neighbor) 
            && !self.tabu_list.is_tabu(neighbor, T::from_f64(self.conf.common.tabu_threshold).unwrap()) {
            Some(self.opt_prob.evaluate(neighbor))
        } else {
            None
        }
    }
}

impl<T: FloatNum> OptimizationAlgorithm<T> for TabuSearch<T>{
    fn step(&mut self) {
        let mut best_neighbor = self.x.clone();
        let mut best_neighbor_fitness = T::neg_infinity();
        
        // Generate and evaluate neighborhood
        let neighbors: Vec<_> = (0..self.conf.common.num_neighbors)
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

            if best_neighbor_fitness > self.st.best_f {
                self.st.best_f = best_neighbor_fitness;
                self.st.best_x = best_neighbor;
                self.iterations_since_improvement = 0;
            } else {
                self.iterations_since_improvement += 1;
            } 
        }

        self.st.pop.set_row(0, &self.x.transpose());
        self.st.fitness[0] = self.opt_prob.evaluate(&self.x);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T> {
        &self.st
    }
} 