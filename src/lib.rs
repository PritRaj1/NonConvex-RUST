pub mod utils;
pub mod continous_ga;
pub mod parallel_tempering;

use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};
use crate::continous_ga::cga::CGA;
use crate::utils::config::{Config};

pub enum OptAlg<T: FloatNum, F: OptProb<T>> {
    CGA(CGA<T, F>),
}

pub struct NonConvexOpt<T: FloatNum, F: OptProb<T>> {
    pub alg: OptAlg<T, F>,
    pub conf: Config,
}

