use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DEConf {
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_f")]
    pub f: f64, // Differential weight
    #[serde(default = "default_cr")]
    pub cr: f64, // Crossover probability
    #[serde(default = "default_strategy")]
    pub strategy: DEStrategy,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum DEStrategy {
    Rand1Bin,   
    Best1Bin,   
    RandToBest1Bin, 
    Best2Bin,    
    Rand2Bin,    
}

fn default_population_size() -> usize { 50 }
fn default_f() -> f64 { 0.8 }
fn default_cr() -> f64 { 0.9 }
fn default_strategy() -> DEStrategy { DEStrategy::Rand1Bin } 