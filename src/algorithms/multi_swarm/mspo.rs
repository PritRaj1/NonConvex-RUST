use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;

use crate::utils::config::MSPOConf;
use crate::algorithms::multi_swarm::swarm::{Swarm, initialize_swarms};
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

pub struct MSPO<T: FloatNum + Send + Sync> {
    pub conf: MSPOConf,
    pub st: State<T>,
    pub swarms: Vec<Swarm<T>>,
    pub opt_prob: OptProb<T>,
}

impl<T: FloatNum + Send + Sync> MSPO<T> {
    pub fn new(conf: MSPOConf, init_pop: DMatrix<T>, opt_prob: OptProb<T>) -> Self {
        let dim = init_pop.ncols();
        let total_particles = init_pop.nrows();
        assert!(total_particles >= conf.num_swarms * conf.swarm_size, 
            "Initial population size must be at least num_swarms * swarm_size");

        let (best_x, best_fitness) = Self::find_best_solution(&init_pop, &opt_prob);

        // Initialize swarms with different regions
        let swarms = initialize_swarms(&conf, dim, &init_pop, &opt_prob);
        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let x = init_pop.row(i).transpose();
                let fit = opt_prob.evaluate(&x);
                let constr = opt_prob.is_feasible(&x);
                (fit, constr)
            })
            .unzip();

        let fitness = DVector::from_vec(fitness);
        let constraints = DVector::from_vec(constraints);

        let st = State {
            best_x,
            best_f: best_fitness,
            pop: init_pop,
            fitness,
            constraints,
            iter: 1
        };

        Self {
            conf,
            st,
            swarms,
            opt_prob,
        }
    }

    fn find_best_solution(
        population: &DMatrix<T>, 
        opt_prob: &OptProb<T>
    ) -> (DVector<T>, T) {
        (0..population.nrows())
            .filter_map(|i| {
                let x = population.row(i).transpose();
                if opt_prob.is_feasible(&x) {
                    Some((x.clone(), opt_prob.evaluate(&x)))
                } else {
                    None
                }
            })
            .max_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap())
            .unwrap_or_else(|| {
                let x = population.row(0).transpose();
                (x.clone(), opt_prob.evaluate(&x))
            })
    }

    fn exchange_information(&mut self) {
        // Collect all best positions and their fitness values
        let best_positions: Vec<_> = self.swarms.iter()
            .map(|swarm| (swarm.global_best_position.clone(), swarm.global_best_fitness))
            .collect();

        // Sort swarms by fitness
        let mut swarm_indices: Vec<_> = (0..self.swarms.len()).collect();
        swarm_indices.sort_by(|&i, &j| best_positions[i].1.partial_cmp(&best_positions[j].1).unwrap());

        // Exchange information between swarms
        self.swarms.par_iter_mut().enumerate().for_each(|(_i, swarm)| {
            let better_swarms: Vec<_> = swarm_indices.iter()
                .filter(|&&idx| best_positions[idx].1 > swarm.global_best_fitness)
                .collect();

            if !better_swarms.is_empty() {
                let num_exchange = (self.conf.swarm_size as f64 * self.conf.exchange_ratio) as usize;
                let mut particles: Vec<_> = swarm.particles.iter_mut()
                    .enumerate()
                    .collect();
                particles.sort_by(|(_, p1), (_, p2)| p1.best_fitness.partial_cmp(&p2.best_fitness).unwrap());

                // Only check constraints when actually updating best positions
                for (_, particle) in particles.iter_mut().take(num_exchange) {
                    for &better_idx in &better_swarms {
                        let (better_pos, better_fitness) = &best_positions[*better_idx];
                        if *better_fitness > particle.best_fitness * (T::one() + T::from_f64(self.conf.improvement_threshold).unwrap()) {
                            // Check feasibility only when we find a potentially better solution
                            if self.opt_prob.is_feasible(better_pos) {
                                particle.best_position = better_pos.clone();
                                particle.best_fitness = *better_fitness;
                                break;
                            }
                        }
                    }
                }
            }
        });
    }

    pub fn get_population(&self) -> DMatrix<T> {
        let total_particles = self.swarms.len() * self.conf.swarm_size;
        let dim = self.st.best_x.len();
        let mut population = DMatrix::zeros(total_particles, dim);
        
        for (swarm_idx, swarm) in self.swarms.iter().enumerate() {
            for (particle_idx, particle) in swarm.particles.iter().enumerate() {
                let row = swarm_idx * self.conf.swarm_size + particle_idx;
                population.set_row(row, &particle.position.transpose());
            }
        }
        
        population
    }
}

impl<T: FloatNum> OptimizationAlgorithm<T> for MSPO<T> {
    fn step(&mut self) {
        // Update each swarm independently
        let results: Vec<_> = self.swarms
            .par_iter_mut()
            .map(|swarm| {
                swarm.update(&self.opt_prob);
                (swarm.global_best_position.clone(), swarm.global_best_fitness)
            })
            .collect();

        for (pos, fitness) in results {
            if fitness > self.st.best_f {
                if self.opt_prob.is_feasible(&pos) {
                    self.st.best_f = fitness;
                    self.st.best_x = pos;
                }
            }
        }

        // Periodic information exchange
        if self.st.iter % self.conf.exchange_interval == 0 {
            self.exchange_information();
        }

        self.st.pop = self.get_population();

        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..self.st.pop.nrows())
            .into_par_iter()
            .map(|i| {
                let x = self.st.pop.row(i).transpose();
                let fit = self.opt_prob.evaluate(&x);
                let constr = self.opt_prob.is_feasible(&x);
                (fit, constr)
            })
            .unzip();

        self.st.fitness = DVector::from_vec(fitness);
        self.st.constraints = DVector::from_vec(constraints);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T> {
        &self.st
    }
} 