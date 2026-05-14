use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use non_convex_opt::utils::opt_prob::{BooleanConstraintFunction, ObjectiveFunction};

#[derive(Clone)]
pub struct Kbf;

impl<D> ObjectiveFunction<f64, D> for Kbf
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<f64, D>) -> f64 {
        let sum_cos4: f64 = x.iter().map(|&xi| xi.cos().powi(4)).sum();
        let prod_cos2: f64 = x.iter().map(|&xi| xi.cos().powi(2)).product();
        let sum_ix2: f64 = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (i as f64 + 1.0) * xi * xi)
            .sum();
        (sum_cos4 - 2.0 * prod_cos2).abs() / sum_ix2.sqrt()
    }

    fn x_lower_bound(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        Some(OVector::<f64, D>::zeros_generic(D::from_usize(x.len()), U1))
    }

    fn x_upper_bound(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        Some(OVector::<f64, D>::from_element_generic(
            D::from_usize(x.len()),
            U1,
            10.0,
        ))
    }
}

// generic gradient-bearing test function for grad-based benches
#[derive(Clone)]
pub struct MultiModalFunction;

impl<D> ObjectiveFunction<f64, D> for MultiModalFunction
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn f(&self, x: &OVector<f64, D>) -> f64 {
        x.iter()
            .map(|&xi| xi.sin() * xi.cos() + 0.1 * xi.powi(2))
            .sum()
    }

    fn gradient(&self, x: &OVector<f64, D>) -> Option<OVector<f64, D>> {
        let mut grad = OVector::<f64, D>::zeros_generic(D::from_usize(x.len()), U1);
        for i in 0..x.len() {
            grad[i] = x[i].cos().powi(2) - x[i].sin().powi(2) + 0.2 * x[i];
        }
        Some(grad)
    }
}

#[derive(Clone)]
pub struct KbfConstraints;

impl<D> BooleanConstraintFunction<f64, D> for KbfConstraints
where
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn g(&self, x: &OVector<f64, D>) -> bool {
        let n = x.len();
        let product: f64 = x.iter().product();
        let sum: f64 = x.iter().sum();
        x.iter().all(|&xi| (-5.0..=5.0).contains(&xi))
            && product > 0.75
            && sum < (15.0 * n as f64) / 2.0
    }
}
