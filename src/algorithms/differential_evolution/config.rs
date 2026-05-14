use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DEConf {
    pub common: CommonConf,
    pub mutation_type: MutationType,
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct CommonConf {}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum MutationType {
    Standard(StandardConf),
    Adaptive(AdaptiveConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct StandardConf {
    #[serde(default = "default_f")]
    pub f: f64,
    #[serde(default = "default_cr")]
    pub cr: f64,
    #[serde(default = "default_strategy")]
    pub strategy: DEStrategy,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdaptiveConf {
    #[serde(default = "default_strategy")]
    pub strategy: DEStrategy,
    #[serde(default = "default_use_jade")]
    pub use_jade: bool,
    #[serde(default = "default_memory_size")]
    pub memory_size: usize,
    // Standard fallback when use_jade=false; JADE ignores these
    #[serde(default = "default_f")]
    pub f: f64,
    #[serde(default = "default_cr")]
    pub cr: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum DEStrategy {
    Rand1Bin,
    Best1Bin,
    RandToBest1Bin,
    Best2Bin,
    Rand2Bin,
}

fn default_f() -> f64 {
    0.8
}
fn default_cr() -> f64 {
    0.9
}
fn default_strategy() -> DEStrategy {
    DEStrategy::Rand1Bin
}
fn default_use_jade() -> bool {
    true
}
fn default_memory_size() -> usize {
    5
}
