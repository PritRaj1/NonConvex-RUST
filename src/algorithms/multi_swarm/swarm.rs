use rand::Rng;
use rayon::prelude::*;
use nalgebra::{
    allocator::Allocator, 
    DefaultAllocator, 
    Dim, 
    OMatrix, 
    OVector,
    U1,
    Dyn,
};

use crate::utils::config::{MSPOConf};
use crate::utils::opt_prob::{FloatNumber as FloatNum, OptProb};

use crate::algorithms::multi_swarm::particle::Particle;

pub struct Swarm<T: FloatNum, D: Dim> 
where 
    DefaultAllocator: Allocator<D> 
                    + Allocator<U1, D>
{
    pub particles: Vec<Particle<T, D>>,   
    pub global_best_position: OVector<T, D>,
    pub global_best_fitness: T,
    pub w: T,
    pub c1: T,
    pub c2: T,
    pub x_min: f64,
    pub x_max: f64,
}

impl<T: FloatNum, D: Dim> Swarm<T, D> 
where 
    T: Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, Dyn, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
                    + Allocator<U1, D>     
                    + Allocator<Dyn, D>
{
    pub fn new(
        num_particles: usize,
        dim: usize,
        w: T,
        c1: T,
        c2: T,
        bounds: (T, T),
        opt_prob: &OptProb<T, D>,
        init_pop: OMatrix<T, Dyn, D>,
    ) -> Self {
        let particles: Vec<_> = (0..num_particles)
            .into_par_iter()
            .map(|i| {
                let mut rng = rand::rng();
                let mut position = OVector::<T, D>::zeros_generic(D::from_usize(dim), U1);
                let fitness;
                
                if i < init_pop.nrows() {
                    // Use initial population if available
                    position = init_pop.row(i).transpose();
                    fitness = opt_prob.evaluate(&position);
                } else {
                    // Generate random position if needed
                    loop {
                        let values = (0..dim).map(|_| {
                            let r = T::from_f64(rng.random::<f64>()).unwrap();
                            bounds.0 + (bounds.1 - bounds.0) * r
                        });
                        let position: OVector<T, D> = OVector::from_iterator_generic(D::from_usize(dim), U1, values);
                        
                        if opt_prob.is_feasible(&position) {
                            fitness = opt_prob.evaluate(&position);
                            break;
                        }
                    }
                }
                
                let values = (0..dim).map(|_| {
                    let r = T::from_f64(rng.random::<f64>()).unwrap();
                    (bounds.1 - bounds.0) * (r - T::from_f64(0.5).unwrap()) * T::from_f64(0.1).unwrap()
                });

                let velocity: OVector<T, D> = OVector::from_iterator_generic(D::from_usize(dim), U1, values);

                Particle::new(position, velocity, fitness)
            })
            .collect();

        let mut best_fitness = T::neg_infinity();
        let mut best_position = OVector::<T, D>::zeros_generic(D::from_usize(dim), U1);

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

    pub fn update(
        &mut self,
        opt_prob: &OptProb<T, D>,
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

pub fn initialize_swarms<T: FloatNum, N: Dim, D: Dim>(
    conf: &MSPOConf,
    dim: usize,
    init_pop: &OMatrix<T, N, D>,
    opt_prob: &OptProb<T, D>
) -> Vec<Swarm<T, D>> 
where 
    T: Send + Sync,
    OVector<T, D>: Send + Sync,
    OMatrix<T, N, D>: Send + Sync,
    DefaultAllocator: Allocator<D>
                    + Allocator<N, D>
                    + Allocator<U1, D>
                    + Allocator<D, D>
{
    let particles_per_swarm = conf.swarm_size;
    let pop_per_swarm = init_pop.nrows() / conf.num_swarms;

    // Find several promising regions 
    let mut promising_centers: Vec<OVector<T, D>> = Vec::new();
    let mut sorted_indices: Vec<usize> = (0..init_pop.nrows()).collect();
    sorted_indices.sort_by(|&i, &j| {
        let fi = opt_prob.evaluate(&init_pop.row(i).transpose());
        let fj = opt_prob.evaluate(&init_pop.row(j).transpose());
        fj.partial_cmp(&fi).unwrap()
    });

    // Select diverse centers from top solutions
    for &idx in sorted_indices.iter().take(conf.num_swarms) {
        let center = init_pop.row(idx).transpose();
        if promising_centers.iter().all(|c| {
            // Ensure centers are sufficiently far apart
            let dist = (c - &center).dot(&(c - &center)).sqrt();
            dist > T::from_f64(0.1 * (conf.x_max - conf.x_min)).unwrap()
        }) {
            promising_centers.push(center);
        }
    }

    // Initialize swarms around these promising regions
    (0..conf.num_swarms)
        .into_par_iter()
        .map(|i| {
            let center = if i < promising_centers.len() {
                promising_centers[i].clone()
            } else {
                // Random center for remaining swarms
                OVector::<T, D>::from_iterator_generic(D::from_usize(dim), U1, (0..dim).map(|_| {
                    T::from_f64(conf.x_min + rand::random::<f64>() * (conf.x_max - conf.x_min)).unwrap()
                }))
            };

            // Initialize particles around the center
            let radius = T::from_f64(0.2 * (conf.x_max - conf.x_min)).unwrap(); // Local search radius
            let start_idx = i * pop_per_swarm;
            let mut swarm_pop: OMatrix<T, Dyn, D> = init_pop.rows(start_idx, particles_per_swarm).into_owned();
            
            // Adjust some particles to be near the center
            for j in 0..particles_per_swarm/2 {
                for k in 0..dim {
                    let r = T::from_f64(rand::random::<f64>()).unwrap();
                    swarm_pop[(j, k)] = center[k] + (r - T::from_f64(0.5).unwrap()) * radius;
                }
            }

            Swarm::new(
                particles_per_swarm,
                dim,
                T::from_f64(conf.w).unwrap(),
                T::from_f64(conf.c1).unwrap(),
                T::from_f64(conf.c2).unwrap(),
                (T::from_f64(conf.x_min).unwrap(), T::from_f64(conf.x_max).unwrap()),
                opt_prob,
                swarm_pop
            )
        })
        .collect()
}