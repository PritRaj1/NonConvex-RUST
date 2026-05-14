use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector, U1};

use crate::utils::config::{AdamConf, OptConf};
use crate::utils::opt_prob::{FloatNumber, OptProb, OptimizationAlgorithm, State};

pub struct Adam<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<N>,
{
    pub conf: AdamConf,
    pub st: State<T, N, D>,
    pub opt_prob: OptProb<T, D>,
    m: OVector<T, D>,
    v: OVector<T, D>,
    v_hat: OVector<T, D>, // AMSGrad max
}

impl<T, N, D> Adam<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N, D> + Allocator<U1, D> + Allocator<N>,
{
    pub fn new(
        conf: AdamConf,
        init_pop: OMatrix<T, N, D>,
        opt_prob: OptProb<T, D>,
        _opt_conf: &OptConf,
        _seed: u64,
    ) -> Self {
        let n = init_pop.ncols();
        let st = State::from_seed(init_pop, &opt_prob);
        let zero = OVector::zeros_generic(D::from_usize(n), U1);

        Self {
            conf,
            st,
            opt_prob,
            m: zero.clone(),
            v: zero.clone(),
            v_hat: zero,
        }
    }
}

impl<T, N, D> OptimizationAlgorithm<T, N, D> for Adam<T, N, D>
where
    T: FloatNumber,
    N: Dim,
    D: Dim,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<N> + Allocator<N, D> + Allocator<U1, D>,
{
    fn step(&mut self) {
        let mut grad = self
            .opt_prob
            .objective
            .gradient(&self.st.best_x)
            .expect("Adam requires gradient");

        if self.conf.weight_decay > 0.0 {
            grad += &self.st.best_x * T::cast(self.conf.weight_decay);
        }

        if self.conf.gradient_clip > 0.0 {
            let clip = T::cast(self.conf.gradient_clip);
            let norm = grad.dot(&grad).sqrt();
            if norm > clip {
                grad *= clip / norm;
            }
        }

        let beta1 = T::cast(self.conf.beta1);
        let beta2 = T::cast(self.conf.beta2);
        self.m = &self.m * beta1 + &grad * (T::one() - beta1);
        self.v = &self.v * beta2 + grad.component_mul(&grad) * (T::one() - beta2);

        let m_hat = &self.m / (T::one() - T::cast(self.conf.beta1.powi(self.st.iter as i32)));
        let v_hat = &self.v / (T::one() - T::cast(self.conf.beta2.powi(self.st.iter as i32)));

        if self.conf.amsgrad {
            self.v_hat = self.v_hat.zip_map(&v_hat, |a, b| a.max(b));
        }

        let lr = T::cast(self.conf.learning_rate);
        let eps = T::cast(self.conf.epsilon);
        let v_denom = if self.conf.amsgrad {
            &self.v_hat
        } else {
            &v_hat
        };
        self.st.best_x += m_hat.component_div(&v_denom.map(|x| x.sqrt() + eps)) * lr;

        if let Some(c) = &self.opt_prob.constraints {
            if !c.g(&self.st.best_x) {
                if let (Some(lb), Some(ub)) = (
                    self.opt_prob.objective.x_lower_bound(&self.st.best_x),
                    self.opt_prob.objective.x_upper_bound(&self.st.best_x),
                ) {
                    self.st.best_x = self
                        .st
                        .best_x
                        .zip_zip_map(&lb, &ub, |x, l, u| x.max(l).min(u));
                }
            }
        }

        let fitness = self.opt_prob.evaluate(&self.st.best_x);
        if fitness > self.st.best_f {
            self.st.best_f = fitness;
        }
        self.st
            .pop
            .row_mut(0)
            .copy_from(&self.st.best_x.transpose());
        self.st.fitness[0] = fitness;
        self.st.constraints[0] = self.opt_prob.is_feasible(&self.st.best_x);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, N, D> {
        &self.st
    }
}
