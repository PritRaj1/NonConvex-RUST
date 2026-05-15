use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TPEConf {
    #[serde(default = "default_n_initial_random")]
    pub n_initial_random: usize,
    // EI candidates sampled from l(x) per step
    #[serde(default = "default_n_candidates")]
    pub n_candidates: usize,
    // top-γ fraction defines the "good" set
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_max_history")]
    pub max_history: usize,
}

impl Default for TPEConf {
    fn default() -> Self {
        Self {
            n_initial_random: default_n_initial_random(),
            n_candidates: default_n_candidates(),
            gamma: default_gamma(),
            max_history: default_max_history(),
        }
    }
}

fn default_n_initial_random() -> usize {
    20
}
fn default_n_candidates() -> usize {
    100
}
fn default_gamma() -> f64 {
    0.25
}
fn default_max_history() -> usize {
    1000
}
