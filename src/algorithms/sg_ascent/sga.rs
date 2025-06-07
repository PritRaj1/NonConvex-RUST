use rand_distr::{Normal, Distribution};
use nalgebra::{
    allocator::Allocator, 
    DefaultAllocator, 
    Dim, 
    OMatrix, 
    OVector,
    U1,
};

use crate::utils::config::SGAConf;
use crate::utils::opt_prob::{
    FloatNumber as FloatNum, 
    OptProb, 
    OptimizationAlgorithm,
    State
};

pub struct SGAscent<T: FloatNum, D: Dim> 
where 
    DefaultAllocator: Allocator<D> 
                     + Allocator<U1, D>
                     + Allocator<U1>
{
    pub conf: SGAConf,
    pub opt_prob: OptProb<T, D>,
    pub x: OVector<T, D>,
    pub st: State<T, U1, D>,
    velocity: OVector<T, D>,
    noise_dist: Normal<f64>,
}

impl<T: FloatNum, D: Dim> SGAscent<T, D> 
where 
    T: Send + Sync,
    DefaultAllocator: Allocator<D> 
                     + Allocator<U1, D>
                     + Allocator<U1>
{
    pub fn new(conf: SGAConf, init_pop: OMatrix<T, U1, D>, opt_prob: OptProb<T, D>) -> Self {
        let init_x: OVector<T, D> = init_pop.row(0).transpose().into_owned();
        let best_f = opt_prob.evaluate(&init_x);
        let noise_dist = Normal::new(0.0, conf.learning_rate).unwrap();
        let n = init_x.len();


        Self {
            conf,
            opt_prob: opt_prob.clone(),
            x: init_x.clone(),
            st: State {
                best_x: init_x.clone(),
                best_f,
                pop: init_pop,
                fitness: OVector::<T, U1>::from_vec(vec![best_f]),
                constraints: OVector::<bool, U1>::from_vec(vec![opt_prob.is_feasible(&init_x.clone())]),
                iter: 1
            },
            velocity: OVector::zeros_generic(D::from_usize(n), U1),
            noise_dist,
        }
    }
}

impl<T: FloatNum, D: Dim> OptimizationAlgorithm<T, U1, D> for SGAscent<T, D> 
where 
    DefaultAllocator: Allocator<D> 
                     + Allocator<U1, D>
                     + Allocator<U1>
{
    fn step(&mut self) {
        let gradient = self.opt_prob.objective.gradient(&self.x).unwrap();
        
        let mut rng = rand::rng();
        let noise = OVector::<T, D>::from_iterator_generic(
            D::from_usize(self.x.len()),
            U1,
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

        self.st.pop.row_mut(0).copy_from(&self.x.transpose());
        self.st.fitness[0] = self.opt_prob.evaluate(&self.x);
        self.st.constraints[0] = self.opt_prob.is_feasible(&self.x);
        self.st.iter += 1;
    }

    fn state(&self) -> &State<T, U1, D> {
        &self.st
    }
} 