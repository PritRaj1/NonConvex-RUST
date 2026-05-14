use crate::utils::opt_prob::{FloatNumber, OptProb};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, U1};
use rand::{rngs::StdRng, Rng};

pub struct Particle<T, D>
where
    T: FloatNumber + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    pub position: OVector<T, D>,
    pub velocity: OVector<T, D>,
    pub best_position: OVector<T, D>,
    pub best_fitness: T,
    rng: StdRng,
}

impl<T, D> Particle<T, D>
where
    T: FloatNumber + Send + Sync,
    D: Dim + Send + Sync,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<U1>,
{
    pub fn new(position: OVector<T, D>, velocity: OVector<T, D>, fitness: T, rng: StdRng) -> Self {
        Self {
            position: position.clone(),
            velocity,
            best_position: position,
            best_fitness: fitness,
            rng,
        }
    }

    pub fn update_velocity_and_position(
        &mut self,
        global_best: &OVector<T, D>,
        w: T,
        c1: T,
        c2: T,
        opt_prob: &OptProb<T, D>,
        bounds: (T, T),
    ) {
        let v_max = (bounds.1 - bounds.0) * T::cast(0.2);
        for i in 0..self.velocity.len() {
            let r1 = T::cast(self.rng.random::<f64>());
            let r2 = T::cast(self.rng.random::<f64>());

            let cognitive = c1 * r1 * (self.best_position[i] - self.position[i]);
            let social = c2 * r2 * (global_best[i] - self.position[i]);

            self.velocity[i] = (w * self.velocity[i] + cognitive + social).clamp(-v_max, v_max);
        }

        // reflective boundary
        let new_positions: Vec<T> = self
            .position
            .iter()
            .zip(self.velocity.iter())
            .map(|(&p, &v)| {
                let np = p + v;
                if np < bounds.0 {
                    (bounds.0 + (bounds.0 - np)).clamp(bounds.0, bounds.1)
                } else if np > bounds.1 {
                    (bounds.1 - (np - bounds.1)).clamp(bounds.0, bounds.1)
                } else {
                    np
                }
            })
            .collect();

        self.position = OVector::<T, D>::from_vec_generic(
            D::from_usize(new_positions.len()),
            U1,
            new_positions,
        );

        let new_fitness = opt_prob.evaluate(&self.position);
        if new_fitness > self.best_fitness && opt_prob.is_feasible(&self.position) {
            self.best_fitness = new_fitness;
            self.best_position = self.position.clone();
        }
    }
}
