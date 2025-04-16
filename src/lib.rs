pub mod utils;
pub mod continous_ga;
pub mod parallel_tempering;
pub mod tabu_search;
pub mod adam;
pub mod grasp;
pub mod sg_ascent;
use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction, BooleanConstraintFunction, OptProb};
use crate::continous_ga::cga::CGA;
use crate::parallel_tempering::pt::PT;
use crate::adam::adam::Adam;
use crate::sg_ascent::sga::SGAscent;
use crate::tabu_search::tabu::TabuSearch;
use crate::grasp::grasp::GRASP;
use crate::utils::config::{Config, AlgConf, OptConf};

pub enum OptAlg<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    CGA(CGA<T, F, G>),
    PT(PT<T, F, G>),
    TS(TabuSearch<T, F, G>),
    Adam(Adam<T, F, G>),
    GRASP(GRASP<T, F, G>),
    SGA(SGAscent<T, F, G>),
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
    pub iter: usize,
    pub converged: bool,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> NonConvexOpt<T, F, G> {
    pub fn new(conf: Config, init_pop: DMatrix<T>, obj_f: F, constr_f: Option<G>) -> Self {
        let opt_prob = OptProb::new(obj_f, constr_f);
        let alg = match conf.alg_conf {
            AlgConf::CGA(cga_conf) => OptAlg::CGA(CGA::new(cga_conf, init_pop, opt_prob)),
            AlgConf::PT(pt_conf) => OptAlg::PT(PT::new(pt_conf, init_pop, opt_prob, conf.opt_conf.max_iter)),
            AlgConf::TS(ts_conf) => OptAlg::TS(TabuSearch::new(ts_conf, init_pop.column(0).into(), opt_prob)),
            AlgConf::Adam(adam_conf) => OptAlg::Adam(Adam::new(adam_conf, init_pop.column(0).into(), opt_prob)),
            AlgConf::GRASP(grasp_conf) => OptAlg::GRASP(GRASP::new(grasp_conf, init_pop.column(0).into(), opt_prob)),
            AlgConf::SGA(sga_conf) => OptAlg::SGA(SGAscent::new(sga_conf, init_pop.column(0).into(), opt_prob)),
        };

        Self { alg, conf: conf.opt_conf, iter: 0, converged: false }
    }

    fn check_convergence(&self, current_best: T, previous_best: T, iter: usize) -> bool {
        let converged = (-current_best).exp() <= T::from_f64(self.conf.atol).unwrap()
            || ((current_best - previous_best).abs() <= T::from_f64(self.conf.rtol).unwrap() && iter > (self.conf.max_iter as f64 * self.conf.rtol_max_iter_fraction).floor() as usize);
        if converged {
            println!("Converged in {} iterations", iter);
        }
        
        converged
    }

    pub fn step(&mut self) {
        if self.converged {
            return;
        }

        let previous_best_fitness = match &self.alg {
            OptAlg::CGA(cga) => cga.best_fitness,
            OptAlg::PT(pt) => pt.best_fitness,
            OptAlg::TS(ts) => ts.best_fitness,
            OptAlg::Adam(adam) => adam.best_fitness,
            OptAlg::GRASP(grasp) => grasp.best_fitness,
            OptAlg::SGA(sga) => sga.best_fitness,
        };

        match &mut self.alg {
            OptAlg::CGA(cga) => cga.step(),
            OptAlg::PT(pt) => pt.step(),
            OptAlg::TS(ts) => ts.step(),
            OptAlg::Adam(adam) => adam.step(),
            OptAlg::GRASP(grasp) => grasp.step(),
            OptAlg::SGA(sga) => sga.step(),
        }

        let current_best_fitness = match &self.alg {
            OptAlg::CGA(cga) => cga.best_fitness,
            OptAlg::PT(pt) => pt.best_fitness,
            OptAlg::TS(ts) => ts.best_fitness,
            OptAlg::Adam(adam) => adam.best_fitness,
            OptAlg::GRASP(grasp) => grasp.best_fitness,
            OptAlg::SGA(sga) => sga.best_fitness,
        };

        self.converged = self.check_convergence(
            current_best_fitness, 
            previous_best_fitness, 
            self.iter
        );
        
        self.iter += 1;
    }
    
    pub fn get_population(&self) -> DMatrix<T> {
        match &self.alg {
            OptAlg::CGA(cga) => cga.population.clone(),
            OptAlg::PT(pt) => pt.population[pt.population.len()-1].clone(),
            OptAlg::TS(ts) => DMatrix::from_columns(&[ts.x.clone()]),
            OptAlg::Adam(adam) => DMatrix::from_columns(&[adam.x.clone()]),
            OptAlg::GRASP(grasp) => DMatrix::from_columns(&[grasp.x.clone()]),
            OptAlg::SGA(sga) => DMatrix::from_columns(&[sga.x.clone()]),
        }
    }

    pub fn get_best_individual(&self) -> DVector<T> {
        match &self.alg {
            OptAlg::CGA(cga) => cga.best_individual.clone(),
            OptAlg::PT(pt) => pt.best_individual.clone(),
            OptAlg::TS(ts) => ts.best_x.clone(),
            OptAlg::Adam(adam) => adam.x.clone(),
            OptAlg::GRASP(grasp) => grasp.best_x.clone(),
            OptAlg::SGA(sga) => sga.x.clone(),
        }
    }

    pub fn run(&mut self) -> Result<T> {
        while !self.converged && self.iter < self.conf.max_iter {
            self.step();
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
            OptAlg::Adam(adam) => {
                let x_matrix = DMatrix::from_columns(&[adam.x.clone()]);
                let fitness_vec = DVector::from_element(x_matrix.ncols(), adam.best_fitness);
                let constraints_vec = DVector::from_element(x_matrix.ncols(), true);
                (adam.x.clone(), adam.best_fitness, x_matrix, fitness_vec, constraints_vec)
            },
            OptAlg::GRASP(grasp) => {
                let x_matrix = DMatrix::from_columns(&[grasp.x.clone()]);
                let fitness_vec = DVector::from_element(x_matrix.ncols(), grasp.best_fitness);
                let constraints_vec = DVector::from_element(x_matrix.ncols(), true);
                (grasp.x.clone(), grasp.best_fitness, x_matrix, fitness_vec, constraints_vec)
            },
            OptAlg::SGA(sga) => {
                let x_matrix = DMatrix::from_columns(&[sga.x.clone()]);
                let fitness_vec = DVector::from_element(x_matrix.ncols(), sga.best_fitness);
                let constraints_vec = DVector::from_element(x_matrix.ncols(), true);
                (sga.x.clone(), sga.best_fitness, x_matrix, fitness_vec, constraints_vec)
            },
        };

        Result {
            best_x,
            best_f,
            final_pop,
            final_fitness,
            final_constraints,
            convergence_iter: self.iter,
        }
    }
}