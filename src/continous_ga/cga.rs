use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::utils::config::CGAConf;
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
        let selector = match conf.selection_method.as_str() {
            "RouletteWheel" => SelectionOperator::RouletteWheel(RouletteWheel::new(conf.population_size, conf.num_parents)),
            "Tournament" => SelectionOperator::Tournament(Tournament::new(conf.population_size, conf.num_parents, conf.tournament_size)),
            "Residual" => SelectionOperator::Residual(Residual::new(conf.population_size, conf.num_parents)),
            _ => panic!("Invalid selection method"),
        };

        let crossover = match conf.crossover_method.as_str() {
            "Random" => CrossoverOperator::Random(Random::new(conf.crossover_prob, conf.population_size)),
            "Heuristic" => CrossoverOperator::Heuristic(Heuristic::new(conf.crossover_prob, conf.population_size)),
            _ => panic!("Invalid crossover method"),
        };

        // Calculate initial fitness and constraints in parallel
        let (fitness, constraints): (Vec<T>, Vec<bool>) = (0..init_pop.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = init_pop.row(i).transpose();
                let fit = opt_prob.objective.f(&individual);
                let constr = opt_prob.constraints.g(&individual)[0];
                (fit, constr)
            })
            .unzip();

        let fitness = DVector::from_vec(fitness);
        let constraints = DVector::from_vec(constraints);

        // Find best individual
        let mut best_idx = 0;
        let mut best_fitness = fitness[0];
        for i in 1..fitness.len() {
            if fitness[i] < best_fitness && constraints[i] {
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
        let selected = self.selector.select(&self.population, &self.fitness, &self.constraints);
        let offspring = self.crossover.crossover(&selected, &self.fitness);
        
        // Evaluate new population in parallel
        let (new_fitness, new_constraints): (Vec<T>, Vec<bool>) = (0..offspring.nrows())
            .into_par_iter()
            .map(|i| {
                let individual = offspring.row(i).transpose();
                let fit = self.opt_prob.objective.f(&individual);
                let constr = self.opt_prob.constraints.g(&individual)[0];
                (fit, constr)
            })
            .unzip();

        let new_fitness = DVector::from_vec(new_fitness);
        let new_constraints = DVector::from_vec(new_constraints);

        // Find best individual
        let mut best_idx = 0;
        let mut best_fitness = new_fitness[0];
        for i in 1..new_fitness.len() {
            if new_fitness[i] < best_fitness && new_constraints[i] {
                best_idx = i;
                best_fitness = new_fitness[i];
            }
        }
        
        // Update best solution if better
        if best_fitness < self.best_fitness {
            self.best_individual = offspring.row(best_idx).transpose();
            self.best_fitness = best_fitness;
        }

        // Update population
        self.population = offspring;
        self.fitness = new_fitness;
        self.constraints = new_constraints;
    }
}