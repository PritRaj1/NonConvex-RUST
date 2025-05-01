use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::alg_conf::de_conf::{DEConf, DEStrategy, MutationType};
use crate::differential_evolution::mutation::*;
use std::collections::VecDeque;

pub struct DE<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: DEConf,
    pub population: DMatrix<T>,
    pub fitness: DVector<T>,
    pub constraints: DVector<bool>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    pub iteration: usize,
    archive: Vec<DVector<T>>,
    archive_fitness: Vec<T>,
    success_history: VecDeque<bool>,
    current_f: f64,
    current_cr: f64,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> DE<T, F, G> {
    pub fn new(conf: DEConf, init_pop: DMatrix<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let population_size = init_pop.nrows();
        let mut fitness = DVector::zeros(population_size);
        let mut constraints = DVector::from_element(population_size, true);
        
        let evaluations: Vec<(T, bool)> = (0..population_size)
            .into_par_iter()
            .map(|i| {
                let x = init_pop.row(i).transpose();
                let fit = opt_prob.objective.f(&x);
                let constr = opt_prob.is_feasible(&x);
                (fit, constr)
            })
            .collect();

        for (i, (fit, constr)) in evaluations.into_iter().enumerate() {
            fitness[i] = fit;
            constraints[i] = constr;
        }

        let mut best_idx = 0;
        let mut best_fitness = fitness[0];
        for i in 1..population_size {
            if fitness[i] > best_fitness && constraints[i] {
                best_idx = i;
                best_fitness = fitness[i];
            }
        }

        // Initialize current F and CR based on mutation type
        let (initial_f, initial_cr) = match &conf.mutation_type {
            MutationType::Standard(standard) => (standard.f, standard.cr),
            MutationType::Adaptive(adaptive) => (
                (adaptive.f_min + adaptive.f_max) / 2.0,
                (adaptive.cr_min + adaptive.cr_max) / 2.0
            ),
        };

        let archive_size = conf.common.archive_size;
        let success_history_size = conf.common.success_history_size;

        Self {
            conf,
            population: init_pop.clone(),
            fitness,
            constraints,
            opt_prob,
            best_x: init_pop.row(best_idx).transpose(),
            best_fitness,
            iteration: 0,
            archive: Vec::with_capacity(archive_size),
            archive_fitness: Vec::with_capacity(archive_size),
            success_history: VecDeque::with_capacity(success_history_size),
            current_f: initial_f,
            current_cr: initial_cr,
        }
    }

    pub fn step(&mut self) {
        let pop_size = self.population.nrows();
        
        let trials: Vec<_> = (0..pop_size)
            .into_par_iter()
            .map(|i| {
                let (trial, trial_fitness, trial_constraint) = self.generate_trial_vector(i);
                
                let success = self.select_trial(
                    trial_fitness,
                    trial_constraint,
                    self.fitness[i],
                    self.constraints[i]
                );

                (i, trial, trial_fitness, trial_constraint, success)
            })
            .collect();

        let mut successes = Vec::new();

        let updates: Vec<_> = trials.into_iter()
            .filter_map(|(i, trial, trial_fitness, trial_constraint, success)| {
                if trial_constraint && trial_fitness > self.fitness[i] {
                    self.update_archive(trial.clone(), trial_fitness);
                }

                successes.push(success);
                
                if success {
                    Some((i, trial, trial_fitness, trial_constraint))
                } else {
                    None
                }
            })
            .collect();

        for success in successes {
            self.success_history.push_back(success);
            if self.success_history.len() > self.conf.common.success_history_size {
                self.success_history.pop_front();
            }
        }

        self.update_parameters();

        let mut new_population = self.population.clone();
        let mut new_fitness = self.fitness.clone();
        let mut new_constraints = self.constraints.clone();

        for (i, trial, trial_fitness, trial_constraint) in updates {
            new_population.set_row(i, &trial.transpose());
            new_fitness[i] = trial_fitness;
            new_constraints[i] = trial_constraint;
        }

        self.population = new_population;
        self.fitness = new_fitness;
        self.constraints = new_constraints;

        for i in 0..pop_size {
            if self.constraints[i] && self.fitness[i] > self.best_fitness {
                self.best_fitness = self.fitness[i];
                self.best_x = self.population.row(i).transpose();
            }
        }

        self.iteration += 1;
    }

    fn generate_trial_vector(&self, target_idx: usize) -> (DVector<T>, T, bool) {
        let strategy = match &self.conf.mutation_type {
            MutationType::Standard(standard) => &standard.strategy,
            MutationType::Adaptive(adaptive) => &adaptive.strategy,
        };

        let strategy: &dyn MutationStrategy<T> = match strategy {
            DEStrategy::Rand1Bin => &Rand1Bin,
            DEStrategy::Best1Bin => &Best1Bin,
            DEStrategy::RandToBest1Bin => &RandToBest1Bin,
            DEStrategy::Best2Bin => &Best2Bin,
            DEStrategy::Rand2Bin => &Rand2Bin,
        };

        let trial = strategy.generate_trial(
            &self.population,
            Some(&self.best_x),
            target_idx,
            T::from_f64(self.current_f).unwrap(),
            T::from_f64(self.current_cr).unwrap(),
        );

        let fitness = self.opt_prob.objective.f(&trial);
        let constraint = self.opt_prob.is_feasible(&trial);

        (trial, fitness, constraint)
    }

    fn update_parameters(&mut self) {
        if let MutationType::Adaptive(adaptive) = &self.conf.mutation_type {
            let success_rate = self.success_history.iter().filter(|&&x| x).count() as f64 
                / self.success_history.len() as f64;
            
            self.current_f = adaptive.f_min + success_rate * (adaptive.f_max - adaptive.f_min);
            self.current_cr = adaptive.cr_min + success_rate * (adaptive.cr_max - adaptive.cr_min);
        }
    }

    fn update_archive(&mut self, x: DVector<T>, fitness: T) {
        if self.archive.len() < self.conf.common.archive_size {
            self.archive.push(x);
            self.archive_fitness.push(fitness);
        } else {
            if let Some(worst_idx) = self.archive_fitness.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
            {
                if fitness > self.archive_fitness[worst_idx] {
                    self.archive[worst_idx] = x;
                    self.archive_fitness[worst_idx] = fitness;
                }
            }
        }
    }

    fn select_trial(
        &self,
        trial_fitness: T,
        trial_constraint: bool,
        current_fitness: T,
        current_constraint: bool,
    ) -> bool {
        match (trial_constraint, current_constraint) {
            (true, true) => {
                // Both feasible - compare fitness with tolerance
                let eps = T::from_f64(1e-10).unwrap();
                trial_fitness > current_fitness + eps
            },
            (true, false) => true,  // Prefer feasible
            (false, true) => false, // Keep feasible
            (false, false) => {
                // Both infeasible - compare fitness
                trial_fitness > current_fitness
            }
        }
    }
} 