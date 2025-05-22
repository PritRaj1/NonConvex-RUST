use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;

use crate::utils::config::NelderMeadConf;
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

pub struct NelderMead<T: FloatNum> {
    pub conf: NelderMeadConf,
    pub st: State<T>,
    pub opt_prob: OptProb<T>,
    pub simplex: Vec<DVector<T>>,
}

impl<T: FloatNum> NelderMead<T> 
where 
    T: Send + Sync 
{
    pub fn new(conf: NelderMeadConf, init_x: DMatrix<T>, opt_prob: OptProb<T>) -> Self {
        let n = init_x.nrows();
        assert_eq!(init_x.ncols(), n + 1, "Initial simplex must have n+1 vertices");
        
        let mut simplex = Vec::with_capacity(n + 1);
        let mut fitness_values = Vec::with_capacity(n + 1);
        
        for j in 0..init_x.ncols() {
            let vertex = init_x.column(j).into_owned();
            simplex.push(vertex.clone());
            fitness_values.push(opt_prob.evaluate(&vertex));
        }
        
        let best_idx = fitness_values.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
            
        Self {
            conf,
            st: State {
                best_x: simplex[best_idx].clone(),
                best_f: fitness_values[best_idx],
                pop: DMatrix::from_columns(&simplex),
                fitness: DVector::from_vec(fitness_values),
                constraints: DVector::from_vec(vec![true; simplex.len()]),
                iter: 1
            },
            opt_prob,
            simplex,
        }
    }

    pub fn centroid(&self, worst_idx: usize) -> DVector<T> {
        let mut centroid = DVector::zeros(self.st.best_x.len());
        for (i, vertex) in self.simplex.iter().enumerate() {
            if i != worst_idx {
                centroid += vertex;
            }
        }
        let scale = T::from_f64((self.simplex.len() - 1) as f64).unwrap();
        &centroid * (T::one() / scale)
    }

    fn get_sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.simplex.len()).collect();
        indices.sort_by(|&i, &j| self.st.fitness[j].partial_cmp(&self.st.fitness[i]).unwrap());
        indices
    }

    fn try_reflection_expansion(&mut self, worst_idx: usize, best_idx: usize, centroid: &DVector<T>) -> bool {
        // Reflect worst point across centroid
        let reflected = centroid + (centroid - &self.simplex[worst_idx]) * T::from_f64(self.conf.alpha).unwrap();
        let reflected_fitness = self.evaluate_point(&reflected);
        
        if reflected_fitness > self.st.fitness[worst_idx] {
            if reflected_fitness > self.st.fitness[best_idx] {
                // Try expansion
                let expanded = centroid + (&reflected - centroid) * T::from_f64(self.conf.gamma).unwrap();
                let expanded_fitness = self.evaluate_point(&expanded);
                
                if expanded_fitness > reflected_fitness {
                    self.update_vertex(worst_idx, expanded, expanded_fitness);
                } else {
                    self.update_vertex(worst_idx, reflected, reflected_fitness);
                }
            } else {
                self.update_vertex(worst_idx, reflected, reflected_fitness);
            }
            return true;
        }

        false
    }

    fn try_contraction(&mut self, worst_idx: usize, _best_idx: usize, centroid: &DVector<T>) -> bool {
        let contracted = centroid + (&self.simplex[worst_idx] - centroid) * T::from_f64(self.conf.rho).unwrap();
        let contracted_fitness = self.evaluate_point(&contracted);
        
        if contracted_fitness > self.st.fitness[worst_idx] {
            self.update_vertex(worst_idx, contracted, contracted_fitness);
            return true;
        }
        
        false
    }

    fn shrink_simplex(&mut self, best_idx: usize) -> bool {
        let best = self.simplex[best_idx].clone();
        let shrink_results: Vec<_> = (0..self.simplex.len()).into_par_iter()
            .filter(|&i| i != best_idx)
            .map(|i| {
                let new_vertex = &best + (&self.simplex[i] - &best) * T::from_f64(self.conf.sigma).unwrap();
                let new_fitness = self.opt_prob.evaluate(&new_vertex);
                (i, new_vertex, new_fitness)
            })
            .collect();
        
        for (i, vertex, fitness) in shrink_results {
            self.update_vertex(i, vertex, fitness);
        }
        true
    }

    // Negativ inf when infeasible
    fn evaluate_point(&self, point: &DVector<T>) -> T {
        if self.opt_prob.is_feasible(point) {
            self.opt_prob.evaluate(point)
        } else {
            T::neg_infinity()
        }
    }

    fn update_vertex(&mut self, idx: usize, vertex: DVector<T>, fitness: T) {
        self.simplex[idx] = vertex;
        self.st.fitness[idx] = fitness;
    }

    fn update_best_solution(&mut self) {
        let best_idx = self.st.fitness.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
            
        self.st.best_x = self.simplex[best_idx].clone();
        if self.st.fitness[best_idx] > self.st.best_f {
            self.st.best_f = self.st.fitness[best_idx];
            self.st.best_x = self.simplex[best_idx].clone();
        }

        self.st.pop = DMatrix::from_columns(&self.simplex);
    }
}

impl<T: FloatNum> OptimizationAlgorithm<T> for NelderMead<T>{
    fn step(&mut self) {
        // Sort vertices by fitness
        let indices = self.get_sorted_indices();
        let (worst_idx, best_idx) = (indices[indices.len() - 1], indices[0]);
        
        // Find centroid excluding worst point
        let centroid = self.centroid(worst_idx);
        
        // Try different operations in sequence until one succeeds
        let _ = self.try_reflection_expansion(worst_idx, best_idx, &centroid) 
            || self.try_contraction(worst_idx, best_idx, &centroid)
            || self.shrink_simplex(best_idx);

        self.update_best_solution();
    }

    fn state(&self) -> &State<T> {
        &self.st
    }
}