use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};
use rand::{rngs::StdRng, Rng};
use std::cmp::Ordering;
use std::collections::VecDeque;

use crate::utils::config::OptConf;
use crate::utils::opt_prob::{FloatNumber, OptProb, OptimizationAlgorithm, State};
use crate::utils::rng;

use crate::algorithms::tpe::config::TPEConf;
use crate::algorithms::tpe::kernels::GaussianKde;

pub struct TPE<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: TPEConf,
    pub st: State<T, N, D>,
    pub opt_prob: OptProb<T, D>,
    observations: VecDeque<(OVector<T, D>, T)>,
    lower: OVector<T, D>,
    upper: OVector<T, D>,
    rng: StdRng,
}

impl<T, N, D> TPE<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync + Clone,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D>,
{
    pub fn new(
        conf: TPEConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        _opt_conf: &OptConf,
        seed: u64,
    ) -> Self {
        let st = State::from_population(init_pop, &opt_prob);
        let dim = st.best_x.len();

        // bounds: whatever objective exposes, else ±10
        let sample = st.best_x.clone();
        let lower = opt_prob.objective.x_lower_bound(&sample).unwrap_or_else(|| {
            OVector::<T, D>::from_element_generic(D::from_usize(dim), U1, T::cast(-10.0))
        });
        let upper = opt_prob.objective.x_upper_bound(&sample).unwrap_or_else(|| {
            OVector::<T, D>::from_element_generic(D::from_usize(dim), U1, T::cast(10.0))
        });

        let mut observations = VecDeque::with_capacity(conf.max_history);
        for i in 0..st.pop.nrows() {
            let x = st.pop.row(i).transpose();
            let f = st.fitness[i];
            observations.push_back((x, f));
        }

        Self {
            conf,
            st,
            opt_prob,
            observations,
            lower,
            upper,
            rng: rng::seeded(seed),
        }
    }

    fn sample_uniform(&mut self) -> OVector<T, D> {
        let dim = self.lower.len();
        let mut out = OVector::<T, D>::zeros_generic(D::from_usize(dim), U1);
        for d in 0..dim {
            let u: f64 = self.rng.random();
            out[d] = self.lower[d] + (self.upper[d] - self.lower[d]) * T::cast(u);
        }
        out
    }

    fn clamp(&self, mut x: OVector<T, D>) -> OVector<T, D> {
        for d in 0..x.len() {
            x[d] = x[d].max(self.lower[d]).min(self.upper[d]);
        }
        x
    }

    // Bergstra-Bengio TPE: argmax_i log l(x_i) − log g(x_i) over candidates drawn from l
    fn propose_tpe(&mut self) -> OVector<T, D> {
        let n = self.observations.len();
        let mut sorted: Vec<(OVector<T, D>, T)> = self.observations.iter().cloned().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let n_good = ((n as f64 * self.conf.gamma).ceil() as usize).clamp(1, n - 1);

        let good_pts: Vec<OVector<T, D>> =
            sorted.iter().take(n_good).map(|(x, _)| x.clone()).collect();
        let bad_pts: Vec<OVector<T, D>> =
            sorted.iter().skip(n_good).map(|(x, _)| x.clone()).collect();

        let min_bw = self.min_bandwidth();
        let l = GaussianKde::fit(good_pts, min_bw);
        let g = GaussianKde::fit(bad_pts, min_bw);

        let mut best_x: Option<OVector<T, D>> = None;
        let mut best_score = T::neg_infinity();
        for _ in 0..self.conf.n_candidates {
            let cand = match l.sample(&mut self.rng) {
                Some(x) => self.clamp(x),
                None => self.sample_uniform(),
            };
            // log EI under TPE ∝ log l(x) − log g(x); empty g returns +∞ score
            let score = l.log_density(&cand) - g.log_density(&cand);
            if score > best_score {
                best_score = score;
                best_x = Some(cand);
            }
        }
        best_x.unwrap_or_else(|| self.sample_uniform())
    }

    // 1e-3 of the longest side; floors KDE to keep log p(x) finite
    fn min_bandwidth(&self) -> T {
        let mut max_range = T::zero();
        for d in 0..self.lower.len() {
            let r = self.upper[d] - self.lower[d];
            if r > max_range {
                max_range = r;
            }
        }
        max_range * T::cast(1e-3)
    }

    fn record(&mut self, x: OVector<T, D>, f: T) {
        self.observations.push_back((x, f));
        while self.observations.len() > self.conf.max_history {
            self.observations.pop_front();
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for TPE<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync + Clone,
    OVector<T, N>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let cand = if self.observations.len() < self.conf.n_initial_random
            || self.observations.len() < 2
        {
            self.sample_uniform()
        } else {
            self.propose_tpe()
        };

        let f = self.opt_prob.evaluate(&cand);
        let feasible = self.opt_prob.is_feasible(&cand);

        if feasible && f > self.st.best_f {
            self.st.best_f = f;
            self.st.best_x = cand.clone();
        }

        self.record(cand, f);

        // mirror most recent rows into st.pop for outer-loop visibility
        let n_rows = self.st.pop.nrows();
        let take = n_rows.min(self.observations.len());
        for i in 0..take {
            let idx = self.observations.len() - take + i;
            let (x, fx) = {
                let entry = &self.observations[idx];
                (entry.0.clone(), entry.1)
            };
            self.st.pop.row_mut(i).copy_from(&x.transpose());
            self.st.fitness[i] = fx;
            self.st.constraints[i] = self.opt_prob.is_feasible(&x);
        }

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}
