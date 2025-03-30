use crate::utils::config::{CGAConf};
use nalgebra::{DVector, DMatrix};
use crate::utils::opt_prob::{FloatNumber as FloatNum, ObjectiveFunction, BooleanConstraintFunction, OptProb};

pub struct CGA<T: FloatNum> {
    pub conf: CGAConf,
    pub population: DMatrix<T>,
    pub fitness: DVector<T>,
    pub constraints: DVector<bool>,
    // pub opt_prob: OptProb<T>,
}