use nalgebra::Scalar;
use nalgebra::DVector;
use num_traits::{Float, FromPrimitive, NumCast, One, Zero};
use simba::scalar::{
    ClosedAdd, ClosedAddAssign, ClosedDiv, 
    ClosedDivAssign, ClosedMul, ClosedMulAssign, 
    ClosedNeg, ClosedSub, ClosedSubAssign, SubsetOf,
};
use std::marker::PhantomData;

// More general trait for float numbers 
pub trait FloatNumber:
    Copy
    + Float
    + NumCast // Convert from other types to float
    + FromPrimitive // Convert from primitive types to float
    + SubsetOf<f64> 
    + Scalar
    + ClosedAdd
    + ClosedMul
    + ClosedDiv
    + ClosedSub
    + ClosedNeg
    + ClosedAddAssign
    + ClosedMulAssign
    + ClosedDivAssign
    + ClosedSubAssign
    + Zero
    + One
    + std::fmt::Debug
    + std::marker::Send
    + std::marker::Sync
    + 'static
{
}

impl FloatNumber for f64 {}
impl FloatNumber for f32 {}

// Trait for objective functions
pub trait ObjectiveFunction<T: FloatNumber>: Send + Sync + Clone {
    fn f(&self, x: &DVector<T>) -> T;
    fn gradient(&self, _x: &DVector<T>) -> Option<DVector<T>> {
        None
    }
    fn x_lower_bound(&self, _x: &DVector<T>) -> Option<DVector<T>> {
        None
    }
    fn x_upper_bound(&self, _x: &DVector<T>) -> Option<DVector<T>> {
        None
    }
}

// Trait for constraint functions
pub trait BooleanConstraintFunction<T: FloatNumber>: Send + Sync + Clone {
    fn g(&self, x: &DVector<T>) -> bool;
}

// Trait for combined optimization problem
#[derive(Clone)]
pub struct OptProb<T: FloatNumber, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub objective: F,
    pub constraints: Option<G>,
    _phantom: PhantomData<T>,
}

impl<T: FloatNumber, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> OptProb<T, F, G> {
    pub fn new(objective: F, constraints: Option<G>) -> Self {
        Self { 
            objective, 
            constraints,
            _phantom: PhantomData,
        }
    }

    pub fn is_feasible(&self, x: &DVector<T>) -> bool {
        match &self.constraints {
            Some(constraints) => constraints.g(x),
            None => true,
        }
    }
}
