use nalgebra::Scalar;
use nalgebra::{DVector, DMatrix};
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
{
}

impl FloatNumber for f64 {}
impl FloatNumber for f32 {}

// Trait for objective functions
pub trait ObjectiveFunction<T: FloatNumber> {
    fn f(&self, x: &DVector<T>) -> T;
    fn df(&self, x: &DVector<T>) -> DVector<T>;
    fn ddf(&self, x: &DVector<T>) -> DMatrix<T>;
}

// Trait for constraint functions
pub trait BooleanConstraintFunction<T: FloatNumber> {
    fn g_single(&self, x: &DVector<T>) -> bool;
    fn g(&self, x: &DVector<T>) -> DVector<bool>;
}

// Trait for combined optimization problem
pub trait OptProb<T: FloatNumber> {
    fn objective(&self) -> &dyn ObjectiveFunction<T>;
    fn constraints(&self) -> &dyn BooleanConstraintFunction<T>;
}



