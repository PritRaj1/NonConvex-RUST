use nalgebra::DVector;
use rand_distr::{Normal, Distribution};
use crate::utils::config::SGAConf;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};

pub struct SGAscent<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> {
    pub conf: SGAConf,
    pub x: DVector<T>,
    pub opt_prob: OptProb<T, F, G>,
    pub best_x: DVector<T>,
    pub best_fitness: T,
    velocity: DVector<T>,
    noise_dist: Normal<f64>,
}

impl<T: FloatNum, F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>> SGAscent<T, F, G> {
    pub fn new(conf: SGAConf, init_x: DVector<T>, opt_prob: OptProb<T, F, G>) -> Self {
        let fitness = opt_prob.objective.f(&init_x);
        let n = init_x.len();
        let noise_dist = Normal::new(0.0, conf.learning_rate).unwrap();
        
        Self {
            conf,
            x: init_x.clone(),
            opt_prob,
            best_x: init_x,
            best_fitness: fitness,
            velocity: DVector::zeros(n),
            noise_dist,
        }
    }

    pub fn step(&mut self) {
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
            let fitness = self.opt_prob.objective.f(&self.x);
            
            if fitness > self.best_fitness {
                self.best_fitness = fitness;
                self.best_x = self.x.clone();
            }
        }
    }
} 