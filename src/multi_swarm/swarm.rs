use nalgebra::{DVector, DMatrix};
use rayon::prelude::*;
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb, ObjectiveFunction, BooleanConstraintFunction};
use crate::multi_swarm::particle::Particle;
use rand::Rng;

pub struct Swarm<T: FloatNum> {
    pub particles: Vec<Particle<T>>,
    pub global_best_position: DVector<T>,
    pub global_best_fitness: T,
    pub w: T,
    pub c1: T,
    pub c2: T,
    pub x_min: f64,
    pub x_max: f64,
}

impl<T: FloatNum> Swarm<T> {
    pub fn new<F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>>(
        num_particles: usize,
        dim: usize,
        w: T,
        c1: T,
        c2: T,
        bounds: (T, T),
        opt_prob: &OptProb<T, F, G>,
        init_pop: DMatrix<T>,
    ) -> Self {
        let particles: Vec<_> = (0..num_particles)
            .into_par_iter()
            .map(|i| {
                let mut rng = rand::rng();
                let mut position;
                let fitness;
                
                if i < init_pop.nrows() {
                    // Use initial population if available
                    position = init_pop.row(i).transpose();
                    fitness = opt_prob.objective.f(&position);
                } else {
                    // Generate random position if needed
                    loop {
                        position = DVector::from_iterator(
                            dim,
                            (0..dim).map(|_| {
                                let r = T::from_f64(rng.random::<f64>()).unwrap();
                                bounds.0 + (bounds.1 - bounds.0) * r
                            })
                        );
                        
                        if opt_prob.is_feasible(&position) {
                            fitness = opt_prob.objective.f(&position);
                            break;
                        }
                    }
                }
                
                let velocity = DVector::from_iterator(
                    dim,
                    (0..dim).map(|_| {
                        let r = T::from_f64(rng.random::<f64>()).unwrap();
                        (bounds.1 - bounds.0) * (r - T::from_f64(0.5).unwrap()) * T::from_f64(0.1).unwrap()
                    })
                );

                Particle::new(position, velocity, fitness)
            })
            .collect();

        let mut best_fitness = T::neg_infinity();
        let mut best_position = DVector::zeros(dim);

        for particle in &particles {
            if particle.best_fitness > best_fitness {
                best_fitness = particle.best_fitness;
                best_position = particle.position.clone();
            }
        }

        Self {
            particles,
            global_best_position: best_position,
            global_best_fitness: best_fitness,
            w,
            c1,
            c2,
            x_min: bounds.0.to_f64().unwrap(),
            x_max: bounds.1.to_f64().unwrap(),
        }
    }

    pub fn update<F: ObjectiveFunction<T>, G: BooleanConstraintFunction<T>>(
        &mut self,
        opt_prob: &OptProb<T, F, G>,
    ) {
        let bounds = (
            T::from_f64(self.x_min).unwrap(),
            T::from_f64(self.x_max).unwrap()
        );

        self.particles.par_iter_mut().for_each(|particle| {
            particle.update_velocity_and_position(
                &self.global_best_position, 
                self.w, 
                self.c1, 
                self.c2,
                opt_prob,
                bounds,
            );
        });

        let best_particle = self.particles.par_iter()
            .reduce_with(|p1, p2| {
                if p1.best_fitness > p2.best_fitness { p1 } else { p2 }
            })
            .unwrap();

        if best_particle.best_fitness > self.global_best_fitness {
            self.global_best_fitness = best_particle.best_fitness;
            self.global_best_position = best_particle.best_position.clone();
        }
    }
} 