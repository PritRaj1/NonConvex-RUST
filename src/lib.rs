pub mod utils;
pub mod continous_ga;
pub mod parallel_tempering;
pub mod tabu_search;

use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction, BooleanConstraintFunction, OptProb};
use crate::continous_ga::cga::CGA;
use crate::parallel_tempering::pt::PT;
use crate::tabu_search::tabu::TabuSearch;
use crate::utils::config::{Config, AlgConf, OptConf};

pub enum OptAlg<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    CGA(CGA<T, F, G>),
    PT(PT<T, F, G>),
    TS(TabuSearch<T, F, G>),
}

pub struct Result<T: FloatNum> {
    pub best_x: DVector<T>,
    pub best_f: T,
    pub final_pop: DMatrix<T>,
    pub final_fitness: DVector<T>,
    pub final_constraints: DVector<bool>,
    pub convergence_iter: usize,
}

pub struct NonConvexOpt<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub alg: OptAlg<T, F, G>,
    pub conf: OptConf,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> NonConvexOpt<T, F, G> {
    pub fn new(conf: Config, init_pop: DMatrix<T>, obj_f: F, constr_f: Option<G>) -> Self {
        let opt_prob = OptProb::new(obj_f, constr_f);
        let alg = match conf.alg_conf {
            AlgConf::CGA(cga_conf) => OptAlg::CGA(CGA::new(cga_conf, init_pop, opt_prob)),
            AlgConf::PT(pt_conf) => OptAlg::PT(PT::new(pt_conf, init_pop, opt_prob, conf.opt_conf.max_iter)),
            AlgConf::TS(ts_conf) => OptAlg::TS(TabuSearch::new(ts_conf, init_pop.column(0).into(), opt_prob)),
        };

        Self { alg, conf: conf.opt_conf }
    }

    pub fn step(&mut self) {
        match &mut self.alg {
            OptAlg::CGA(cga) => cga.step(),
            OptAlg::PT(pt) => pt.step(),
            OptAlg::TS(ts) => ts.step(),
        }
    }
    
    pub fn get_population(&self) -> DMatrix<T> {
        match &self.alg {
            OptAlg::CGA(cga) => cga.population.clone(),
            OptAlg::PT(pt) => pt.population[pt.population.len()-1].clone(),
            OptAlg::TS(ts) => DMatrix::from_columns(&[ts.x.clone()]),
        }
    }

    pub fn get_best_individual(&self) -> DVector<T> {
        match &self.alg {
            OptAlg::CGA(cga) => cga.best_individual.clone(),
            OptAlg::PT(pt) => pt.best_individual.clone(),
            OptAlg::TS(ts) => ts.best_x.clone(),
        }
    }

    pub fn run(&mut self) -> Result<T> {
        let mut previous_best_fitness = T::infinity();
        let mut iter = 0;
        
        for _ in 0..self.conf.max_iter {
            self.step();
            match &mut self.alg {
                OptAlg::CGA(cga) => {
                    if (-cga.best_fitness).exp() <= T::from_f64(self.conf.atol).unwrap() || (cga.best_fitness - previous_best_fitness).abs() <= T::from_f64(self.conf.rtol).unwrap() {
                        println!("Converged in {} iterations", iter);
                        break;
                    }
                    previous_best_fitness = cga.best_fitness;
                    iter += 1;
                },
                OptAlg::PT(pt) => {
                    if (-pt.best_fitness).exp() <= T::from_f64(self.conf.atol).unwrap() || (pt.best_fitness - previous_best_fitness).abs() <= T::from_f64(self.conf.rtol).unwrap() {
                        println!("Converged in {} iterations", iter);
                        break;
                    }
                    previous_best_fitness = pt.best_fitness;
                    iter += 1;
                },
                OptAlg::TS(ts) => {
                    if (-ts.best_fitness).exp() <= T::from_f64(self.conf.atol).unwrap() || (ts.best_fitness - previous_best_fitness).abs() <= T::from_f64(self.conf.rtol).unwrap() {
                        println!("Converged in {} iterations", iter);
                        break;
                    }
                    previous_best_fitness = ts.best_fitness;
                    iter += 1;
                }
            }
        }

        let (best_x, best_f, final_pop, final_fitness, final_constraints) = match &self.alg {
            OptAlg::CGA(cga) => (cga.best_individual.clone(), cga.best_fitness, cga.population.clone(), cga.fitness.clone(), cga.constraints.clone()),
            OptAlg::PT(pt) => (pt.best_individual.clone(), pt.best_fitness, pt.population[pt.population.len()-1].clone(), pt.fitness[pt.fitness.len()-1].clone(), pt.constraints[pt.constraints.len()-1].clone()),
            OptAlg::TS(ts) => {
                let x_matrix = DMatrix::from_columns(&[ts.x.clone()]);
                let fitness_vec = DVector::from_element(x_matrix.ncols(), ts.best_fitness);
                let constraints_vec = DVector::from_element(x_matrix.ncols(), true);
                (ts.best_x.clone(), ts.best_fitness, x_matrix, fitness_vec, constraints_vec)
            },
        };

        Result {
            best_x,
            best_f,
            final_pop,
            final_fitness,
            final_constraints,
            convergence_iter: iter,
        }
    }
}