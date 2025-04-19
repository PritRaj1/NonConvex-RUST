use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::utils::config::{CGAConf, CrossoverConf, SelectionConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::continous_ga::selection::*;
use crate::continous_ga::crossover::*;


pub struct CGA<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: CGAConf,
    pub population: DMatrix<T>,
    pub fitness: DVector<T>,
    pub constraints: DVector<bool>,
    pub selector: SelectionOperator,
    pub crossover: CrossoverOperator,
    pub opt_prob: OptProb<T, F, G>,
    pub best_individual: DVector<T>,
    pub best_fitness: T,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> CGA<T, F, G> {
    pub fn new(conf: CGAConf, init_pop: DMatrix<T>, opt_prob: OptProb<T, F, G>,) -> Self {
        let selector = match &conf.selection {
            SelectionConf::RouletteWheel(_) => SelectionOperator::RouletteWheel(RouletteWheel::new(conf.common.population_size, conf.common.num_parents)),
            SelectionConf::Tournament(tournament) => SelectionOperator::Tournament(Tournament::new(conf.common.population_size, conf.common.num_parents, tournament.tournament_size)),
            SelectionConf::Residual(_) => SelectionOperator::Residual(Residual::new(conf.common.population_size, conf.common.num_parents)),
        };

        let crossover = match &conf.crossover {
            CrossoverConf::Random(random) => CrossoverOperator::Random(Random::new(random.crossover_prob, conf.common.population_size)),
            CrossoverConf::Heuristic(heuristic) => CrossoverOperator::Heuristic(Heuristic::new(heuristic.crossover_prob, conf.common.population_size)),
        };
        
        // Calculate initial fitness and constraints in parallel
        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = init_pop.row(i).transpose();
                let fit = opt_prob.objective.f(&individual);
                let constr = opt_prob.is_feasible(&individual);
                (fit, constr)
            })
            .unzip();

        let fitness = DVector::from_vec(fitness);
        let constraints = DVector::from_vec(constraints);

        // Find best individual
        let mut best_idx = 0;
        let mut best_fitness = fitness[0];
        for i in 1..fitness.len() {
            if fitness[i] > best_fitness && constraints[i] {
                best_idx = i;
                best_fitness = fitness[i];
            }
        }
        let best_individual = init_pop.row(best_idx).transpose();

        Self { 
            conf,
            population: init_pop,
            fitness,
            constraints,
            selector,
            crossover,
            opt_prob,
            best_individual,
            best_fitness,
        }
    }

    pub fn step(&mut self) {
        // Select parents
        let selected = self.selector.select(&self.population, &self.fitness, &self.constraints);
        
        // Create offspring through crossover
        let mut offspring = self.crossover.crossover(&selected, &self.fitness);

        // Evaluate offspring
        let (new_fitness, new_constraints): (Vec<T>, Vec<bool>) = (0..offspring.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = offspring.row(i).transpose();
                let fit = self.opt_prob.objective.f(&individual);
                let constr = self.opt_prob.is_feasible(&individual);
                (fit, constr)
            })
            .unzip();

        let mut new_fitness = DVector::from_vec(new_fitness);
        let mut new_constraints = DVector::from_vec(new_constraints);

        // Elitism: Keep the best individual from previous generation
        let mut best_old_idx = 0;
        let mut best_old_fitness = self.fitness[0];
        for i in 1..self.fitness.len() {
            if self.fitness[i] > best_old_fitness && self.constraints[i] {
                best_old_idx = i;
                best_old_fitness = self.fitness[i];
            }
        }

        // Replace worst offspring with best old individual if better
        let mut worst_new_idx = 0;
        let mut worst_new_fitness = new_fitness[0];
        for i in 1..new_fitness.len() {
            if new_fitness[i] < worst_new_fitness {
                worst_new_idx = i;
                worst_new_fitness = new_fitness[i];
            }
        }

        if best_old_fitness > worst_new_fitness {
            offspring.set_row(worst_new_idx, &self.population.row(best_old_idx));
            new_fitness[worst_new_idx] = best_old_fitness;
            new_constraints[worst_new_idx] = self.constraints[best_old_idx];
        }

        // Update population and metrics
        self.population = offspring;
        self.fitness = new_fitness;
        self.constraints = new_constraints;

        // Update best solution
        for i in 0..self.fitness.len() {
            if self.fitness[i] > self.best_fitness && self.constraints[i] {
                self.best_fitness = self.fitness[i];
                self.best_individual = self.population.row(i).transpose();
            }
        }
    }
}