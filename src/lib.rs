pub mod utils;
pub mod continous_ga;
pub mod parallel_tempering;

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::continous_ga::cga::CGA;
use crate::parallel_tempering::pt::PT;
use crate::utils::config::{Config};

pub enum OptAlg<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    CGA(CGA<T, F, G>),
    PT(PT<T, F, G>),
}

pub struct NonConvexOpt<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub alg: OptAlg<T, F, G>,
    pub conf: Config,
}

