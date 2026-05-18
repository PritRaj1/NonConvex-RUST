use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};

use crate::utils::config::{BacktrackingConf, GoldenSectionConf, StrongWolfeConf};
use crate::utils::opt_prob::{FloatNumber, OptProb};

pub trait LineSearch<T, D>
where
    T: FloatNumber,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T;
}

pub struct BacktrackingLineSearch {
    conf: BacktrackingConf,
}

impl BacktrackingLineSearch {
    pub fn new(conf: &BacktrackingConf) -> Self {
        Self { conf: conf.clone() }
    }
}

// Armijo-only (sufficient-increase) backtracking
impl<T, D> LineSearch<T, D> for BacktrackingLineSearch
where
    T: FloatNumber,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let c1 = T::cast(self.conf.c1);
        let rho = T::cast(self.conf.rho);
        let gp = g.dot(p);
        let mut alpha = T::one();
        // cap to avoid pathological shrinking
        for _ in 0..64 {
            let x_new = x + p * alpha;
            if opt_prob.evaluate(&x_new) >= f + c1 * alpha * gp {
                return alpha;
            }
            alpha *= rho;
        }
        alpha
    }
}

pub struct StrongWolfeLineSearch {
    conf: StrongWolfeConf,
}

impl StrongWolfeLineSearch {
    pub fn new(conf: &StrongWolfeConf) -> Self {
        Self { conf: conf.clone() }
    }
}

// Strong-Wolfe for ascent: ∇f·p > 0
//   sufficient increase : f(x+αp) ≥ f + c1 α ∇f·p
//   strong curvature    : |∇f(x+αp)·p| ≤ c2 |∇f·p|
impl<T, D> LineSearch<T, D> for StrongWolfeLineSearch
where
    T: FloatNumber,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        f: T,
        g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let c1 = T::cast(self.conf.c1);
        let c2 = T::cast(self.conf.c2);
        let initial_gp = g.dot(p);
        // bad direction: not ascent → fall back to zero step
        if initial_gp <= T::zero() {
            return T::zero();
        }
        let curvature_rhs = c2 * initial_gp;

        let mut alpha = T::one();
        let mut alpha_lo = T::zero();
        let mut alpha_hi: Option<T> = None;
        let alpha_max = T::cast(1024.0);

        for _ in 0..self.conf.max_iters {
            let x_new = x + p * alpha;
            let f_new = opt_prob.evaluate(&x_new);
            let g_new_p = opt_prob.objective.gradient(&x_new).unwrap().dot(p);
            let abs_g_new_p = if g_new_p >= T::zero() {
                g_new_p
            } else {
                T::zero() - g_new_p
            };

            if f_new < f + c1 * alpha * initial_gp || g_new_p < T::zero() {
                // step too large or overshot peak → tighten upper bracket
                alpha_hi = Some(alpha);
            } else if abs_g_new_p <= curvature_rhs {
                return alpha;
            } else {
                // step too small → raise lower bracket
                alpha_lo = alpha;
            }

            alpha = match alpha_hi {
                Some(ah) => (alpha_lo + ah) / T::cast(2.0),
                None => {
                    let next = alpha * T::cast(2.0);
                    if next > alpha_max {
                        alpha_max
                    } else {
                        next
                    }
                }
            };
        }
        alpha
    }
}

pub struct GoldenSectionLineSearch {
    conf: GoldenSectionConf,
}

impl GoldenSectionLineSearch {
    pub fn new(conf: &GoldenSectionConf) -> Self {
        Self { conf: conf.clone() }
    }

    fn bracket_maximum<T: FloatNumber, D: Dim>(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> (T, T, T)
    where
        DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
    {
        let bracket_factor = T::cast(self.conf.bracket_factor);
        let mut a = T::zero();
        let mut b = T::one();
        let mut c = b * bracket_factor;
        let mut fa = opt_prob.evaluate(&(x + p * a));
        let mut fb = opt_prob.evaluate(&(x + p * b));
        let mut fc = opt_prob.evaluate(&(x + p * c));
        // expand until fa < fb && fc < fb (peak straddled)
        for _ in 0..64 {
            if fb >= fa && fb >= fc {
                break;
            }
            if fc > fb {
                a = b;
                fa = fb;
                b = c;
                fb = fc;
                c = b * bracket_factor;
                fc = opt_prob.evaluate(&(x + p * c));
            } else {
                c = b;
                fc = fb;
                b = a;
                fb = fa;
                a = b / bracket_factor;
                fa = opt_prob.evaluate(&(x + p * a));
            }
        }
        (a, b, c)
    }
}

impl<T, D> LineSearch<T, D> for GoldenSectionLineSearch
where
    T: FloatNumber,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    fn search(
        &self,
        x: &OVector<T, D>,
        p: &OVector<T, D>,
        _f: T,
        _g: &OVector<T, D>,
        opt_prob: &OptProb<T, D>,
    ) -> T {
        let resphi = T::cast((3.0_f64 - (5.0_f64).sqrt()) / 2.0);
        let tol = T::cast(self.conf.tol);
        let (mut a, b, mut c) = self.bracket_maximum(x, p, opt_prob);
        let mut x0 = b - resphi * (c - a);
        let mut x1 = a + resphi * (c - a);
        let mut f0 = opt_prob.evaluate(&(x + p * x0));
        let mut f1 = opt_prob.evaluate(&(x + p * x1));

        for _ in 0..self.conf.max_iters {
            if num_traits::Float::abs(c - a) < tol {
                break;
            }
            if f0 > f1 {
                c = x1;
                x1 = x0;
                f1 = f0;
                x0 = b - resphi * (c - a);
                f0 = opt_prob.evaluate(&(x + p * x0));
            } else {
                a = x0;
                x0 = x1;
                f0 = f1;
                x1 = a + resphi * (c - a);
                f1 = opt_prob.evaluate(&(x + p * x1));
            }
        }
        (a + c) / T::cast(2.0)
    }
}
