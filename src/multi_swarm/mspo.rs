use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::config::MSPOConf;
use crate::multi_swarm::swarm::Swarm;
use rayon::prelude::*;
use rand;

pub struct MSPO<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: MSPOConf,
    pub swarms: Vec<Swarm<T>>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    iteration: usize,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> MSPO<T, F, G> {
    pub fn new(conf: MSPOConf, init_pop: DMatrix<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let dim = init_pop.ncols();
        let total_particles = init_pop.nrows();
        assert!(total_particles >= conf.num_swarms * conf.swarm_size, 
            "Initial population size must be at least num_swarms * swarm_size");

        // Find best solution from initial population
        let (best_x, best_fitness) = Self::find_best_solution(&init_pop, &opt_prob);

        // Initialize swarms with different regions
        let swarms = Self::initialize_swarms(&conf, dim, &init_pop, &opt_prob);

        Self {
            conf,
            swarms,
            opt_prob,
            best_x,
            best_fitness,
            iteration: 0,
        }
    }

    fn initialize_swarms(
        conf: &MSPOConf,
        dim: usize,
        init_pop: &DMatrix<T>,
        opt_prob: &OptProb<T, F, G>
    ) -> Vec<Swarm<T>> {
        let particles_per_swarm = conf.swarm_size;
        let pop_per_swarm = init_pop.nrows() / conf.num_swarms;

        // Find several promising regions 
        let mut promising_centers: Vec<DVector<T>> = Vec::new();
        let mut sorted_indices: Vec<usize> = (0..init_pop.nrows()).collect();
        sorted_indices.sort_by(|&i, &j| {
            let fi = opt_prob.objective.f(&init_pop.row(i).transpose());
            let fj = opt_prob.objective.f(&init_pop.row(j).transpose());
            fj.partial_cmp(&fi).unwrap()
        });

        // Select diverse centers from top solutions
        for &idx in sorted_indices.iter().take(conf.num_swarms) {
            let center = init_pop.row(idx).transpose();
            if promising_centers.iter().all(|c| {
                // Ensure centers are sufficiently far apart
                let dist = (c - &center).dot(&(c - &center)).sqrt();
                dist > T::from_f64(0.1 * (conf.x_max - conf.x_min)).unwrap()
            }) {
                promising_centers.push(center);
            }
        }

        // Initialize swarms around these promising regions
        (0..conf.num_swarms)
            .into_par_iter()
            .map(|i| {
                let center = if i < promising_centers.len() {
                    promising_centers[i].clone()
                } else {
                    // Random center for remaining swarms
                    DVector::from_iterator(dim, (0..dim).map(|_| {
                        T::from_f64(conf.x_min + rand::random::<f64>() * (conf.x_max - conf.x_min)).unwrap()
                    }))
                };

                // Initialize particles around the center
                let radius = T::from_f64(0.2 * (conf.x_max - conf.x_min)).unwrap(); // Local search radius
                let start_idx = i * pop_per_swarm;
                let mut swarm_pop = init_pop.rows(start_idx, particles_per_swarm).into_owned();
                
                // Adjust some particles to be near the center
                for j in 0..particles_per_swarm/2 {
                    for k in 0..dim {
                        let r = T::from_f64(rand::random::<f64>()).unwrap();
                        swarm_pop[(j, k)] = center[k] + (r - T::from_f64(0.5).unwrap()) * radius;
                    }
                }

                Swarm::new(
                    particles_per_swarm,
                    dim,
                    T::from_f64(conf.w).unwrap(),
                    T::from_f64(conf.c1).unwrap(),
                    T::from_f64(conf.c2).unwrap(),
                    (T::from_f64(conf.x_min).unwrap(), T::from_f64(conf.x_max).unwrap()),
                    opt_prob,
                    swarm_pop
                )
            })
            .collect()
    }

    fn find_best_solution(
        population: &DMatrix<T>, 
        opt_prob: &OptProb<T, F, G>
    ) -> (DVector<T>, T) {
        (0..population.nrows())
            .filter_map(|i| {
                let x = population.row(i).transpose();
                if opt_prob.is_feasible(&x) {
                    Some((x.clone(), opt_prob.objective.f(&x)))
                } else {
                    None
                }
            })
            .max_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap())
            .unwrap_or_else(|| {
                let x = population.row(0).transpose();
                (x.clone(), opt_prob.objective.f(&x))
            })
    }

    pub fn step(&mut self) {
        // Update each swarm independently
        let results: Vec<_> = self.swarms
            .par_iter_mut()
            .map(|swarm| {
                swarm.update(&self.opt_prob);
                (swarm.global_best_position.clone(), swarm.global_best_fitness)
            })
            .collect();

        // Update global best - only check constraints for potential new best
        for (pos, fitness) in results {
            if fitness > self.best_fitness {
                if self.opt_prob.is_feasible(&pos) {
                    self.best_fitness = fitness;
                    self.best_x = pos;
                }
            }
        }

        // Periodic information exchange
        if self.iteration % self.conf.exchange_interval == 0 {
            self.exchange_information();
        }

        self.iteration += 1;
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
        let dim = self.best_x.len();
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