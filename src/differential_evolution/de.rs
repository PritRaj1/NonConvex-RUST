use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::alg_conf::de_conf::{DEConf, DEStrategy};
use crate::differential_evolution::mutation::*;

pub struct DE<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: DEConf,
    pub population: DMatrix<T>,
    pub fitness: DVector<T>,
    pub constraints: DVector<bool>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    pub iteration: usize,
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

        Self {
            conf,
            population: init_pop.clone(),
            fitness,
            constraints,
            opt_prob,
            best_x: init_pop.row(best_idx).transpose(),
            best_fitness,
            iteration: 0,
        }
    }

    pub fn step(&mut self) {
        let pop_size = self.population.nrows();
        let mut new_population = self.population.clone();
        let mut new_fitness = self.fitness.clone();
        let mut new_constraints = self.constraints.clone();

        let updates: Vec<(usize, DVector<T>, T, bool)> = (0..pop_size)
            .into_par_iter()
            .filter_map(|i| {
                let (trial, trial_fitness, trial_constraint) = self.generate_trial_vector(i);
                
                // Replace if trial is better (greedy selection)
                if trial_constraint && trial_fitness > self.fitness[i] {
                    Some((i, trial, trial_fitness, trial_constraint))
                } else {
                    None
                }
            })
            .collect();

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
        let f = T::from_f64(self.conf.f).unwrap();
        let cr = T::from_f64(self.conf.cr).unwrap();
        
        let strategy: &dyn MutationStrategy<T> = match self.conf.strategy {
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
            f,
            cr,
        );

        let fitness = self.opt_prob.objective.f(&trial);
        let constraint = self.opt_prob.is_feasible(&trial);

        (trial, fitness, constraint)
    }
} 