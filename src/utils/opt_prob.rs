use nalgebra::{Scalar, DVector, DMatrix};
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

pub trait CloneBox<T: FloatNumber> {
    fn clone_box(&self) -> Box<dyn ObjectiveFunction<T>>;
}

pub trait CloneBoxConstraint<T: FloatNumber> {
    fn clone_box_constraint(&self) -> Box<dyn BooleanConstraintFunction<T>>;
}

impl<T: FloatNumber, F: ObjectiveFunction<T> + Clone + 'static> CloneBox<T> for F {
    fn clone_box(&self) -> Box<dyn ObjectiveFunction<T>> {
        Box::new(self.clone())
    }
}

impl<T: FloatNumber, F: BooleanConstraintFunction<T> + Clone + 'static> CloneBoxConstraint<T> for F {
    fn clone_box_constraint(&self) -> Box<dyn BooleanConstraintFunction<T>> {
        Box::new(self.clone())
    }
}

pub trait ObjectiveFunction<T: FloatNumber>: CloneBox<T> + Send + Sync {
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

pub trait BooleanConstraintFunction<T: FloatNumber>: CloneBoxConstraint<T> + Send + Sync {
    fn g(&self, x: &DVector<T>) -> bool;
}

impl<T: FloatNumber> Clone for Box<dyn ObjectiveFunction<T>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl<T: FloatNumber> Clone for Box<dyn BooleanConstraintFunction<T>> {
    fn clone(&self) -> Self {
        self.clone_box_constraint()
    }
}

#[derive(Clone)]
pub struct OptProb<T: FloatNumber> {
    pub objective: Box<dyn ObjectiveFunction<T>>,
    pub constraints: Option<Box<dyn BooleanConstraintFunction<T>>>,
}

impl<T: FloatNumber> OptProb<T> {
    pub fn new(
        objective: Box<dyn ObjectiveFunction<T>>,
        constraints: Option<Box<dyn BooleanConstraintFunction<T>>>,
    ) -> Self {
        Self {
            objective,
            constraints,
        }
    }

    pub fn is_feasible(&self, x: &DVector<T>) -> bool {
        match &self.constraints {
            Some(constraints) => constraints.g(x),
            None => true,
        }
    }

    pub fn evaluate(&self, x: &DVector<T>) -> T {
        self.objective.f(x)
    }
}

pub struct State<T: FloatNumber> {
    pub best_x: DVector<T>,
    pub best_f: T,
    pub pop: DMatrix<T>,
    pub fitness: DVector<T>,
    pub constraints: DVector<bool>,
    pub iter: usize,
}

pub trait OptimizationAlgorithm<T: FloatNumber>: Send + Sync {
    fn step(&mut self);
    fn state(&self) -> &State<T>;
}

