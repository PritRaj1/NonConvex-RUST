use rand::{rngs::StdRng, Rng};

use crate::utils::rng;

pub enum SwapCheck {
    Periodic(Periodic),
    Stochastic(Stochastic),
    Always,
}

impl SwapCheck {
    pub fn should_swap(&mut self, step: usize) -> bool {
        match self {
            SwapCheck::Periodic(p) => p.should_swap(step),
            SwapCheck::Stochastic(s) => s.should_swap(),
            SwapCheck::Always => true,
        }
    }
}

pub struct Periodic {
    pub swap_frequency: f64,
    pub total_steps: usize,
}

impl Periodic {
    pub fn new(swap_frequency: f64, total_steps: usize) -> Self {
        Self {
            swap_frequency,
            total_steps,
        }
    }

    pub fn should_swap(&self, current_step: usize) -> bool {
        current_step.is_multiple_of((self.swap_frequency * self.total_steps as f64) as usize)
    }
}

pub struct Stochastic {
    pub swap_probability: f64,
    rng: StdRng,
}

impl Stochastic {
    pub fn new(swap_probability: f64, seed: u64) -> Self {
        Self {
            swap_probability,
            rng: rng::seeded(seed),
        }
    }

    pub fn should_swap(&mut self) -> bool {
        self.rng.random::<f64>() < self.swap_probability
    }
}
