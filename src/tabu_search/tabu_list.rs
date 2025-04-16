use nalgebra::DVector;
use std::collections::VecDeque;
use crate::utils::opt_prob::FloatNumber as FloatNum;
use crate::utils::config::TabuConf;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum TabuType {
    Standard,
    Reactive {
        min_size: usize,
        max_size: usize,
        increase_factor: f64,
        decrease_factor: f64,
    },
}

impl From<&TabuConf> for TabuType {
    fn from(conf: &TabuConf) -> Self {
        match conf.tabu_type.as_str() {
            "Reactive" => TabuType::Reactive {
                min_size: conf.min_tabu_size,
                max_size: conf.max_tabu_size,
                increase_factor: conf.increase_factor,
                decrease_factor: conf.decrease_factor,
            },
            _ => TabuType::Standard,
        }
    }
}

pub struct TabuList<T: FloatNum> {
    items: VecDeque<DVector<T>>,
    max_size: usize,
    tabu_type: TabuType,
}

impl<T: FloatNum> TabuList<T> {
    pub fn new(max_size: usize, tabu_type: TabuType) -> Self {
        Self {
            items: VecDeque::with_capacity(max_size),
            max_size,
            tabu_type,
        }
    }

    pub fn is_tabu(&self, x: &DVector<T>, threshold: T) -> bool {
        self.items.par_iter().any(|tabu_x| {
            let diff = x - tabu_x;
            diff.dot(&diff).sqrt() < threshold
        })
    }

    pub fn update(&mut self, x: DVector<T>, iterations_since_improvement: usize) {
        match self.tabu_type {
            TabuType::Reactive { min_size, max_size, increase_factor, decrease_factor } => {
                let new_size = if iterations_since_improvement > 10 {
                    ((self.items.len() as f64) * decrease_factor) as usize
                } else {
                    ((self.items.len() as f64) * increase_factor) as usize
                };
                self.max_size = new_size.clamp(min_size, max_size);
            },
            TabuType::Standard => {}
        }

        self.items.push_front(x);
        while self.items.len() > self.max_size {
            self.items.pop_back();
        }
    }
}