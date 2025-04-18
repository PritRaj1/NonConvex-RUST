use nalgebra::DVector;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use rand::Rng;

pub struct Particle<T: FloatNum> {
    pub position: DVector<T>,
    pub velocity: DVector<T>,
    pub best_position: DVector<T>,
    pub best_fitness: T,
}

impl<T: FloatNum> Particle<T> {
    pub fn new(position: DVector<T>, velocity: DVector<T>, fitness: T) -> Self {
        Self {
            position: position.clone(),
            velocity,
            best_position: position,
            best_fitness: fitness,
        }
    }

    pub fn update_velocity_and_position<F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>>(
        &mut self,
        global_best: &DVector<T>,
        w: T,
        c1: T,
        c2: T,
        opt_prob: &OptProb<T, F, G>,
        bounds: (T, T),
    ) {
        let mut rng = rand::rng();
        
        // Update velocity
        for i in 0..self.velocity.len() {
            let r1 = T::from_f64(rng.random::<f64>()).unwrap();
            let r2 = T::from_f64(rng.random::<f64>()).unwrap();
            
            let cognitive = c1 * r1 * (self.best_position[i] - self.position[i]);
            let social = c2 * r2 * (global_best[i] - self.position[i]);
            
            // Add velocity clamping
            let v_max = (bounds.1 - bounds.0) * T::from_f64(0.1).unwrap();
            self.velocity[i] = (w * self.velocity[i] + cognitive + social).clamp(-v_max, v_max);
        }

        // Update position with bounds checking only
        let new_positions: Vec<T> = self.position
            .iter()
            .zip(self.velocity.iter())
            .map(|(&p, &v)| {
                let new_pos = p + v;
                new_pos.clamp(bounds.0, bounds.1)
            })
            .collect();

        let final_position = DVector::from_vec(new_positions);
        self.position = final_position;
        
        // Only update best position if new position is better AND feasible
        let new_fitness = opt_prob.objective.f(&self.position);
        if new_fitness > self.best_fitness && opt_prob.is_feasible(&self.position) {
            self.best_fitness = new_fitness;
            self.best_position = self.position.clone();
        }
    }
} 