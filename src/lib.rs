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
}