use nalgebra::{
    allocator::Allocator, 
    DefaultAllocator, 
    Dim, 
    OMatrix, 
    OVector,
    U1,
};

use crate::utils::config::AdamConf;
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

pub struct Adam<T: FloatNum, D: Dim> 
where 
    DefaultAllocator: Allocator<D> 
                     + Allocator<U1, D>
                     + Allocator<U1>
{
    pub conf: AdamConf,
    pub st: State<T, U1, D>,
    pub opt_prob: OptProb<T, D>,
    m: OVector<T, D>,  // First moment estimate
    v: OVector<T, D>,  // Second moment estimate
}

impl<T: FloatNum, D: Dim> Adam<T, D> 
where 
    DefaultAllocator: Allocator<D> 
                     + Allocator<U1, D>
                     + Allocator<U1>
{
    pub fn new(conf: AdamConf, init_pop: OMatrix<T, U1, D>, opt_prob: OptProb<T, D>) -> Self {
        let init_x: OVector<T, D> = init_pop.row(0).transpose().into_owned();
        let best_f = opt_prob.evaluate(&init_x);
        let n = init_x.len();

        Self {
            conf,
            st: State {
                best_x: init_x.clone(),
                best_f,
                pop: init_pop,
                fitness: OVector::<T, U1>::from_vec(vec![best_f]),
                constraints: OVector::<bool, U1>::from_vec(vec![opt_prob.is_feasible(&init_x.clone())]),
                iter: 1
            },
            opt_prob,
            m: OVector::zeros_generic(D::from_usize(n), U1),
            v: OVector::zeros_generic(D::from_usize(n), U1),
        }
    }
}

impl<T: FloatNum, D: Dim> OptimizationAlgorithm<T, U1, D> for Adam<T, D> 
where 
    DefaultAllocator: Allocator<D> 
                     + Allocator<U1, D>
                     + Allocator<U1>
{
    fn step(&mut self) {
        let grad = self.opt_prob.objective.gradient(&self.st.best_x)
            .expect("ADAM requires gradient information");
                
        // Biased moment estimates
        self.m = self.m.clone() * T::from_f64(self.conf.beta1).unwrap() + 
                 grad.clone() * T::from_f64(1.0 - self.conf.beta1).unwrap();
        self.v = self.v.clone() * T::from_f64(self.conf.beta2).unwrap() + 
                 grad.component_mul(&grad) * T::from_f64(1.0 - self.conf.beta2).unwrap();
        
        // Bias-corrected moment estimates
        let m_hat = self.m.clone() / 
            (T::one() - T::from_f64(self.conf.beta1.powi(self.st.iter as i32)).unwrap());
        let v_hat = self.v.clone() / 
            (T::one() - T::from_f64(self.conf.beta2.powi(self.st.iter as i32)).unwrap());
        
        let step_size = T::from_f64(self.conf.learning_rate).unwrap();
        let epsilon = T::from_f64(self.conf.epsilon).unwrap();
        
        let update = m_hat.component_div(&v_hat.map(|x| x.sqrt() + epsilon)) * step_size;
        self.st.best_x += update;

        // Clamp onto feasible set 
        if let Some(ref constraints) = self.opt_prob.constraints {
            if !constraints.g(&self.st.best_x) {
                if let (Some(lb), Some(ub)) = (self.opt_prob.objective.x_lower_bound(&self.st.best_x), 
                                             self.opt_prob.objective.x_upper_bound(&self.st.best_x)) {
                    for i in 0..self.st.best_x.len() {
                        self.st.best_x[i] = self.st.best_x[i].max(lb[i]).min(ub[i]);
                    }
                }
            }
        }

        let fitness = self.opt_prob.evaluate(&self.st.best_x);
        if fitness > self.st.best_f {
            self.st.best_f = fitness;
            self.st.best_x = self.st.best_x.clone();
        }

        self.st.pop.row_mut(0).copy_from(&self.st.best_x.transpose());
        self.st.fitness[0] = fitness;
        self.st.constraints[0] = self.opt_prob.is_feasible(&self.st.best_x);

        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, U1, D> {
        &self.st
    }
} 