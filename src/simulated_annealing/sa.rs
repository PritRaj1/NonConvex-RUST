use nalgebra::DVector;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::alg_conf::sa_conf::SAConf;
use crate::simulated_annealing::neighbor_gen::GaussianGenerator;
use crate::simulated_annealing::cooling::{CoolingSchedule, ExponentialCooling};
use crate::simulated_annealing::acceptance::MetropolisAcceptance;
use rayon::prelude::*;

pub struct SimulatedAnnealing<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: SAConf,
    pub opt_prob: OptProb<T, F, G>,
    pub x: DVector<T>,
    pub fitness: T,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    pub temperature: T,
    pub iteration: usize,
    no_improve_count: usize,
    neighbor_gen: GaussianGenerator<T, F, G>,
    cooling_schedule: ExponentialCooling,
    acceptance: MetropolisAcceptance<T, F, G>,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> SimulatedAnnealing<T, F, G> {
    pub fn new(conf: SAConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let fitness = opt_prob.objective.f(&init_x);
        let opt_prob_clone = opt_prob.clone();
        let opt_prob_clone2 = opt_prob.clone();
        
        Self {
            conf: conf.clone(),
            x: init_x.clone(),
            opt_prob,
            best_x: init_x,
            best_fitness: fitness,
            fitness,
            temperature: T::from_f64(conf.initial_temp).unwrap(),
            iteration: 0,
            no_improve_count: 0,
            neighbor_gen: GaussianGenerator::new(opt_prob_clone, T::from_f64(conf.step_size).unwrap()),
            cooling_schedule: ExponentialCooling,
            acceptance: MetropolisAcceptance::new(opt_prob_clone2),
        }
    }

    pub fn step(&mut self) {
        let min_step = self.conf.step_size * 0.01;
        let step_size = (self.conf.step_size * 
            (self.temperature / T::from_f64(self.conf.initial_temp).unwrap())
                .to_f64()
                .unwrap()
                .sqrt())
            .max(min_step);
        
        let bounds = (
            T::from_f64(self.conf.x_min).unwrap(),
            T::from_f64(self.conf.x_max).unwrap()
        );

        // Generate and evaluate neighbors in parallel
        let neighbors: Vec<_> = (0..self.conf.num_neighbors)
            .into_par_iter()
            .map(|_| self.neighbor_gen.generate(&self.x, step_size, bounds, self.temperature))
            .collect();

        let mut improved = false;
        for neighbor in neighbors {
            let neighbor_fitness = self.opt_prob.objective.f(&neighbor);
            
            if neighbor_fitness > self.best_fitness && self.opt_prob.is_feasible(&neighbor) {
                self.best_fitness = neighbor_fitness;
                self.best_x = neighbor.clone();
                self.no_improve_count = 0;
                improved = true;
            }

            // Use Metropolis criterion for current solution
            if self.acceptance.accept(
                &self.x,
                self.fitness, 
                &neighbor,
                neighbor_fitness,
                self.temperature,
                T::from_f64(step_size).unwrap()) && self.opt_prob.is_feasible(&neighbor)
            {
                self.x = neighbor;
                self.fitness = neighbor_fitness;
            }
        }

        if !improved {
            self.no_improve_count += 1;
        }

        // More aggressive reheating if stuck for a long time
        if self.no_improve_count > self.conf.reheat_after * 2 {
            self.temperature = self.cooling_schedule.reheat(T::from_f64(self.conf.initial_temp).unwrap());
            self.no_improve_count = 0;
            self.x = self.best_x.clone(); // Reset to best known solution
        }
        
        self.iteration += 1;
    }
} 