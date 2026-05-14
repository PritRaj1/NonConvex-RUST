use crate::utils::rng;
use rand::{rngs::StdRng, Rng};
use rand_distr::{Distribution, Normal};

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub enum ParameterAdaptationType {
    JADE(Box<JADEParameterAdaptation>),
    Standard(StandardParameterAdaptation),
}

impl ParameterAdaptationType {
    pub fn generate_parameters(&mut self) -> (f64, f64) {
        match self {
            ParameterAdaptationType::JADE(jade) => jade.generate_jade_parameters(),
            ParameterAdaptationType::Standard(standard) => standard.get_parameters(),
        }
    }

    pub fn record_success(&mut self, f: f64, cr: f64) {
        match self {
            ParameterAdaptationType::JADE(jade) => jade.record_success(f, cr),
            ParameterAdaptationType::Standard(_) => {
                // Standard adaptation doesn't track individual successes
            }
        }
    }

    pub fn update_parameters(&mut self) {
        match self {
            ParameterAdaptationType::JADE(jade) => jade.update_memory(),
            ParameterAdaptationType::Standard(_) => {
                // Standard adaptation updates are handled externally
            }
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub struct JADEParameterAdaptation {
    pub f_memory: Vec<f64>,
    pub cr_memory: Vec<f64>,
    pub memory_pointer: usize,
    pub successful_f: Vec<f64>,
    pub successful_cr: Vec<f64>,
    pub rng: StdRng,
}

impl JADEParameterAdaptation {
    pub fn new(memory_size: usize, seed: u64) -> Self {
        let mut rng = rng::seeded(seed);
        let f_memory = (0..memory_size)
            .map(|_| rng.random::<f64>() * 0.5 + 0.25)
            .collect();
        let cr_memory = (0..memory_size)
            .map(|_| rng.random::<f64>() * 0.5 + 0.25)
            .collect();

        Self {
            f_memory,
            cr_memory,
            memory_pointer: 0,
            successful_f: Vec::new(),
            successful_cr: Vec::new(),
            rng,
        }
    }

    // JADE: F ~ Cauchy(μ_F, 0.1), CR ~ Normal(μ_CR, 0.1)
    pub fn generate_jade_parameters(&mut self) -> (f64, f64) {
        let memory_idx = self.rng.random_range(0..self.f_memory.len());

        let mu_f = self.f_memory[memory_idx];
        let f = loop {
            let u: f64 = self.rng.random();
            let cand = mu_f + 0.1 * (std::f64::consts::PI * (u - 0.5)).tan();
            if cand > 0.0 {
                break cand.min(1.0);
            }
        };

        let mu_cr = self.cr_memory[memory_idx];
        let normal = Normal::new(mu_cr, 0.1).expect("Normal stddev > 0");
        let cr = normal.sample(&mut self.rng).clamp(0.0, 1.0);

        (f, cr)
    }

    pub fn record_success(&mut self, f: f64, cr: f64) {
        self.successful_f.push(f);
        self.successful_cr.push(cr);
    }

    pub fn update_memory(&mut self) {
        if self.successful_f.is_empty() || self.successful_cr.is_empty() {
            return;
        }

        // Update mem using Lehmer mean (arithmetic mean is more sensitive to outliers)
        let f_mean = self.lehmer_mean(&self.successful_f);
        let cr_mean = self.lehmer_mean(&self.successful_cr);

        // Update at current pointer
        self.f_memory[self.memory_pointer] = f_mean;
        self.cr_memory[self.memory_pointer] = cr_mean;

        // Move pointer
        self.memory_pointer = (self.memory_pointer + 1) % self.f_memory.len();
        self.successful_f.clear();
        self.successful_cr.clear();
    }

    fn lehmer_mean(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.5;
        }
        let sum_squares: f64 = values.iter().map(|&x| x * x).sum();
        let sum: f64 = values.iter().sum();
        if sum.abs() < 1e-10 {
            0.5
        } else {
            sum_squares / sum
        }
    }
}

#[derive(Clone)]
pub struct StandardParameterAdaptation {
    pub current_f: f64,
    pub current_cr: f64,
}

impl StandardParameterAdaptation {
    pub fn new(f: f64, cr: f64) -> Self {
        Self {
            current_f: f,
            current_cr: cr,
        }
    }

    pub fn get_parameters(&self) -> (f64, f64) {
        (self.current_f, self.current_cr)
    }
}
