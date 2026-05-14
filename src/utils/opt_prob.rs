use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, Scalar};
use num_traits::{Float, FromPrimitive};
use simba::scalar::{
    ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedNeg, ClosedSubAssign, SubsetOf,
};

pub trait FloatNumber:
    Float
    + FromPrimitive
    + SubsetOf<f64>
    + Scalar
    + ClosedAddAssign
    + ClosedMulAssign
    + ClosedDivAssign
    + ClosedSubAssign
    + ClosedNeg
    + Send
    + Sync
    + 'static
{
    // method-call form so inference resolves at the call site
    #[inline]
    fn cast(x: f64) -> Self {
        <Self as FromPrimitive>::from_f64(x).expect("f64 -> Self cast failed")
    }
}

impl FloatNumber for f64 {}
impl FloatNumber for f32 {}

pub trait CloneBox<T: FloatNumber, D: Dim> {
    fn clone_box(&self) -> Box<dyn ObjectiveFunction<T, D>>;
}

pub trait CloneBoxConstraint<T: FloatNumber, D: Dim> {
    fn clone_box_constraint(&self) -> Box<dyn BooleanConstraintFunction<T, D>>;
}

impl<T: FloatNumber, D: Dim, F: ObjectiveFunction<T, D> + Clone + 'static> CloneBox<T, D> for F
where
    DefaultAllocator: Allocator<D>,
{
    fn clone_box(&self) -> Box<dyn ObjectiveFunction<T, D>> {
        Box::new(self.clone())
    }
}

impl<T: FloatNumber, D: Dim, F: BooleanConstraintFunction<T, D> + Clone + 'static>
    CloneBoxConstraint<T, D> for F
where
    DefaultAllocator: Allocator<D>,
{
    fn clone_box_constraint(&self) -> Box<dyn BooleanConstraintFunction<T, D>> {
        Box::new(self.clone())
    }
}

pub trait ObjectiveFunction<T: FloatNumber, D: Dim>: CloneBox<T, D> + Send + Sync
where
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<T, D>) -> T;

    fn gradient(&self, _x: &OVector<T, D>) -> Option<OVector<T, D>> {
        None
    }

    fn x_lower_bound(&self, _x: &OVector<T, D>) -> Option<OVector<T, D>> {
        None
    }

    fn x_upper_bound(&self, _x: &OVector<T, D>) -> Option<OVector<T, D>> {
        None
    }
}

pub trait BooleanConstraintFunction<T: FloatNumber, D: Dim>:
    CloneBoxConstraint<T, D> + Send + Sync
where
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<T, D>) -> bool;
}

pub struct OptProb<T: FloatNumber, D: Dim>
where
    DefaultAllocator: Allocator<D>,
{
    pub objective: Box<dyn ObjectiveFunction<T, D>>,
    pub constraints: Option<Box<dyn BooleanConstraintFunction<T, D>>>,
}

impl<T: FloatNumber, D: Dim> OptProb<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(
        objective: Box<dyn ObjectiveFunction<T, D>>,
        constraints: Option<Box<dyn BooleanConstraintFunction<T, D>>>,
    ) -> Self {
        Self {
            objective,
            constraints,
        }
    }

    pub fn is_feasible(&self, x: &OVector<T, D>) -> bool {
        match &self.constraints {
            Some(constraints) => constraints.g(x),
            None => true,
        }
    }

    pub fn evaluate(&self, x: &OVector<T, D>) -> T {
        self.objective.f(x)
    }
}

impl<T, D> Clone for OptProb<T, D>
where
    T: FloatNumber,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn clone(&self) -> Self {
        Self {
            objective: self.objective.clone_box(),
            constraints: self.constraints.as_ref().map(|c| c.clone_box_constraint()),
        }
    }
}

#[derive(Clone)]
pub struct State<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    pub best_x: OVector<T, D>,
    pub best_f: T,
    pub pop: OMatrix<T, N, D>,
    pub fitness: OVector<T, N>,
    pub constraints: OVector<bool, N>,
    pub iter: usize,
}

impl<T, N, D> State<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<nalgebra::U1, D>,
{
    // seed from row 0 of `init_pop`; broadcast its fitness/feasibility to all rows
    pub fn from_seed(init_pop: OMatrix<T, N, D>, opt_prob: &OptProb<T, D>) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let feasible = opt_prob.is_feasible(&init_x);
        let n = init_pop.nrows();
        Self {
            best_x: init_x,
            best_f,
            pop: init_pop,
            fitness: OVector::from_element_generic(N::from_usize(n), nalgebra::U1, best_f),
            constraints: OVector::from_element_generic(N::from_usize(n), nalgebra::U1, feasible),
            iter: 1,
        }
    }

    // evaluate every row, pick the best feasible
    pub fn from_population(init_pop: OMatrix<T, N, D>, opt_prob: &OptProb<T, D>) -> Self {
        let n = init_pop.nrows();
        let fitness = OVector::from_iterator_generic(
            N::from_usize(n),
            nalgebra::U1,
            (0..n).map(|i| opt_prob.evaluate(&init_pop.row(i).transpose())),
        );
        let constraints = OVector::from_iterator_generic(
            N::from_usize(n),
            nalgebra::U1,
            (0..n).map(|i| opt_prob.is_feasible(&init_pop.row(i).transpose())),
        );
        let best_idx = (0..n)
            .filter(|&i| constraints[i])
            .max_by(|&a, &b| {
                fitness[a]
                    .partial_cmp(&fitness[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        Self {
            best_x: init_pop.row(best_idx).transpose(),
            best_f: fitness[best_idx],
            pop: init_pop,
            fitness,
            constraints,
            iter: 1,
        }
    }
}

pub trait OptimizationAlgorithm<T: FloatNumber, N: Dim, D: Dim>
where
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D>,
{
    fn step(&mut self);
    fn state(&self) -> &State<T, N, D>;
    fn get_simplex(&self) -> Option<&Vec<OVector<T, D>>> {
        None
    }
    fn get_replica_populations(&self) -> Option<Vec<OMatrix<T, N, D>>> {
        None
    }
    fn get_replica_temperatures(&self) -> Option<Vec<T>> {
        None
    }
}
