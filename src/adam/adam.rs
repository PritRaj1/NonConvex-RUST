use nalgebra::DVector;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::utils::config::AdamConf;

pub struct Adam<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: AdamConf,
    pub x: DVector<T>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    m: DVector<T>,  // First moment estimate
    v: DVector<T>,  // Second moment estimate
    t: usize,       // Timestep
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> Adam<T, F, G> {
    pub fn new(conf: AdamConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let fitness = opt_prob.objective.f(&init_x);
        let dim = init_x.len();
        
        Self {
            conf,
            x: init_x.clone(),
            opt_prob,
            best_x: init_x,
            best_fitness: fitness,
            m: DVector::zeros(dim),
            v: DVector::zeros(dim),
            t: 0,
        }
    }

    pub fn step(&mut self) {
        let grad = self.opt_prob.objective.gradient(&self.x)
            .expect("ADAM requires gradient information");
        
        self.t += 1;
        
        // Biased moment estimates
        self.m = self.m.clone() * T::from_f64(self.conf.beta1).unwrap() + 
                 grad.clone() * T::from_f64(1.0 - self.conf.beta1).unwrap();
        self.v = self.v.clone() * T::from_f64(self.conf.beta2).unwrap() + 
                 grad.component_mul(&grad) * T::from_f64(1.0 - self.conf.beta2).unwrap();
        
        // Bias-corrected moment estimates
        let m_hat = self.m.clone() / 
            (T::one() - T::from_f64(self.conf.beta1.powi(self.t as i32)).unwrap());
        let v_hat = self.v.clone() / 
            (T::one() - T::from_f64(self.conf.beta2.powi(self.t as i32)).unwrap());
        
        let step_size = T::from_f64(self.conf.learning_rate).unwrap();
        let epsilon = T::from_f64(self.conf.epsilon).unwrap();
        
        let update = m_hat.component_div(&v_hat.map(|x| x.sqrt() + epsilon)) * step_size;
        self.x += update;

        // Clamp onto feasible set if needed
        if let Some(ref constraints) = self.opt_prob.constraints {
            if !constraints.g(&self.x) {
                if let (Some(lb), Some(ub)) = (self.opt_prob.objective.x_lower_bound(&self.x), 
                                             self.opt_prob.objective.x_upper_bound(&self.x)) {
                    for i in 0..self.x.len() {
                        self.x[i] = self.x[i].max(lb[i]).min(ub[i]);
                    }
                }
            }
        }

        let fitness = self.opt_prob.objective.f(&self.x);
        if fitness > self.best_fitness {
            self.best_fitness = fitness;
            self.best_x = self.x.clone();
        }
    }
} 