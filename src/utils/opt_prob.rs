use nalgebra::Scalar;
use nalgebra::DVector;
use num_traits::{Float, FromPrimitive, NumCast, One, Zero};
use simba::scalar::{
    ClosedAdd, ClosedAddAssign, ClosedDiv, 
    ClosedDivAssign, ClosedMul, ClosedMulAssign, 
    ClosedNeg, ClosedSub, ClosedSubAssign, SubsetOf,
};

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
pub trait ObjectiveFunction<T: FloatNumber>: Send + Sync {
    fn f(&self, x: &DVector<T>) -> T;
    fn gradient(&self, _x: &DVector<T>) -> Option<DVector<T>> {
        None
    }
}

// Trait for constraint functions
pub trait BooleanConstraintFunction<T: FloatNumber>: Send + Sync {
    fn g(&self, x: &DVector<T>) -> DVector<bool>;
}

// Trait for combined optimization problem
pub trait OptProb<T: FloatNumber>: Send + Sync {
    fn objective(&self, x: &DVector<T>) -> T;
    fn constraints(&self, x: &DVector<T>) -> DVector<bool> {
        DVector::from_element(x.len(), true)
    }
}



