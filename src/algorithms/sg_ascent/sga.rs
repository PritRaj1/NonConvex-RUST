use nalgebra::{DVector, DMatrix};
use rand_distr::{Normal, Distribution};

use crate::utils::config::SGAConf;
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

pub struct SGAscent<T: FloatNum> {
    pub conf: SGAConf,
    pub opt_prob: OptProb<T>,
    pub x: DVector<T>,
    pub st: State<T>,
    velocity: DVector<T>,
    noise_dist: Normal<f64>,
}

impl<T: FloatNum> SGAscent<T> 
where 
    T: Send + Sync 
{
    pub fn new(conf: SGAConf, init_pop: DMatrix<T>, opt_prob: OptProb<T>) -> Self {
        let init_x = init_pop.row(0).transpose();
        let best_f = opt_prob.evaluate(&init_x);
        let noise_dist = Normal::new(0.0, conf.learning_rate).unwrap();
        let n = init_x.len();


        Self {
            conf,
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            st: State {
                best_x: init_x.clone(),
                best_f: best_f,
                pop: DMatrix::from_columns(&[init_x.clone()]),
                fitness: vec![best_f].into(),
                constraints: vec![opt_prob.is_feasible(&init_x)].into(),
                iter: 1
            },
            velocity: DVector::zeros(n),
            noise_dist,
        }
    }
}

impl<T: FloatNum> OptimizationAlgorithm<T> for SGAscent<T> {
    fn step(&mut self) {
        let gradient = self.opt_prob.objective.gradient(&self.x).unwrap();
        
        let mut rng = rand::rng();
        let noise = DVector::from_iterator(
            self.x.len(),
            (0..self.x.len()).map(|_| T::from_f64(self.noise_dist.sample(&mut rng)).unwrap())
        );
        
        let noisy_gradient = gradient + noise;
        
        self.velocity = &self.velocity * T::from_f64(self.conf.momentum).unwrap() + 
                       &noisy_gradient * T::from_f64(self.conf.learning_rate).unwrap();

        self.x += &self.velocity;
        
        if self.opt_prob.is_feasible(&self.x) {
            let fitness = self.opt_prob.evaluate(&self.x);
            
            if fitness > self.st.best_f {
                self.st.best_f = fitness;
                self.st.best_x = self.x.clone();
            }
        }

        self.st.pop.set_row(0, &self.x.transpose());
        self.st.fitness = DVector::from(vec![self.opt_prob.evaluate(&self.x)]);
        self.st.constraints = DVector::from(vec![self.opt_prob.is_feasible(&self.x)]);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T> {
        &self.st
    }
} 