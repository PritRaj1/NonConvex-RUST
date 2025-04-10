pub mod utils;
pub mod continous_ga;
pub mod parallel_tempering;

use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction, BooleanConstraintFunction, OptProb};
use crate::continous_ga::cga::CGA;
use crate::parallel_tempering::pt::PT;
use crate::utils::config::{Config, AlgConf, OptConf};

pub enum OptAlg<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    CGA(CGA<T, F, G>),
    PT(PT<T, F, G>),
}

pub struct Result<T: FloatNum> {
    pub best_x: DVector<T>,
    pub best_f: T,
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
        };

        Self { alg, conf: conf.opt_conf }
    }

    pub fn run(&mut self) -> Result<T> {

        let mut previous_best_fitness = T::infinity();
        
        for _ in 0..self.conf.max_iter {
            match &mut self.alg {
                OptAlg::CGA(cga) => {
                    cga.step();
                    if cga.best_fitness <= T::from_f64(self.conf.atol).unwrap() || (cga.best_fitness - previous_best_fitness).abs() <= T::from_f64(self.conf.rtol).unwrap() {
                        break;
                    }
                    previous_best_fitness = cga.best_fitness;
                },
                OptAlg::PT(pt) => {
                    pt.step();
                    if pt.best_fitness <= T::from_f64(self.conf.atol).unwrap() || (pt.best_fitness - previous_best_fitness).abs() <= T::from_f64(self.conf.rtol).unwrap() {
                        break;
                    }
                    previous_best_fitness = pt.best_fitness;
                },
            }
        }

        let (best_x, best_f) = match &self.alg {
            OptAlg::CGA(cga) => (cga.best_individual.clone(), cga.best_fitness),
            OptAlg::PT(pt) => (pt.best_individual.clone(), pt.best_fitness),
        };

        Result {
            best_x,
            best_f,
        }
    }
}