use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use rand::{rngs::StdRng, Rng};
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

use crate::utils::opt_prob::FloatNumber;

// Gaussian KDE with per-dim Silverman bandwidth
pub struct GaussianKde<T, D>
where
    T: FloatNumber,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub points: Vec<OVector<T, D>>,
    pub bandwidth: OVector<T, D>,
}

impl<T, D> GaussianKde<T, D>
where
    T: FloatNumber,
    D: Dim,
    OVector<T, D>: Clone,
    DefaultAllocator: Allocator<D> + Allocator<U1, D>,
{
    // Silverman per-dim: h_d = 1.06·σ_d·n^(-1/5), floored to min_bw
    pub fn fit(points: Vec<OVector<T, D>>, min_bw: T) -> Self {
        let n = points.len();
        let dim = if n == 0 { 0 } else { points[0].len() };
        let mut bandwidth = OVector::<T, D>::from_element_generic(D::from_usize(dim), U1, min_bw);
        if n < 2 || dim == 0 {
            return Self { points, bandwidth };
        }

        let inv_n = T::one() / T::cast(n as f64);
        for d in 0..dim {
            let mut mean = T::zero();
            for p in &points {
                mean += p[d];
            }
            mean *= inv_n;
            let mut var = T::zero();
            for p in &points {
                let diff = p[d] - mean;
                var += diff * diff;
            }
            var /= T::cast((n - 1).max(1) as f64);
            let sigma = num_traits::Float::sqrt(var);
            let factor = T::cast(1.06 * (n as f64).powf(-0.2));
            let h = sigma * factor;
            bandwidth[d] = if h > min_bw { h } else { min_bw };
        }
        Self { points, bandwidth }
    }

    // log p(x): −log n + logsumexp_i [−½ Σ_d ((x_d−μ_id)/h_d)²] − Σ_d log(h_d·√(2π))
    pub fn log_density(&self, x: &OVector<T, D>) -> T {
        if self.points.is_empty() {
            return T::neg_infinity();
        }
        let dim = x.len();
        let mut log_norm = T::zero();
        for d in 0..dim {
            log_norm += num_traits::Float::ln(self.bandwidth[d] * T::cast((2.0 * PI).sqrt()));
        }

        let mut log_terms: Vec<T> = Vec::with_capacity(self.points.len());
        for p in &self.points {
            let mut q = T::zero();
            for d in 0..dim {
                let z = (x[d] - p[d]) / self.bandwidth[d];
                q += z * z;
            }
            log_terms.push(T::cast(-0.5) * q);
        }
        let max_log = log_terms
            .iter()
            .copied()
            .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });
        if !num_traits::Float::is_finite(max_log) {
            return T::neg_infinity();
        }
        let mut sum_exp = T::zero();
        for t in log_terms {
            sum_exp += num_traits::Float::exp(t - max_log);
        }
        let log_n = num_traits::Float::ln(T::cast(self.points.len() as f64));
        max_log + num_traits::Float::ln(sum_exp) - log_n - log_norm
    }

    // x = μ_i + h ⊙ ξ, ξ~N(0,I_d), i uniform
    pub fn sample(&self, rng: &mut StdRng) -> Option<OVector<T, D>> {
        if self.points.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..self.points.len());
        let center = &self.points[idx];
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut out = center.clone();
        for d in 0..out.len() {
            let noise: f64 = normal.sample(rng);
            out[d] = center[d] + self.bandwidth[d] * T::cast(noise);
        }
        Some(out)
    }
}
