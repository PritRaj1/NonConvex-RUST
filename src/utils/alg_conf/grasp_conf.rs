use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GRASPConf {
    #[serde(default = "default_grasp_num_candidates")]
    pub num_candidates: usize,
    #[serde(default = "default_grasp_alpha")]
    pub alpha: f64,
    #[serde(default = "default_grasp_num_neighbors")]
    pub num_neighbors: usize,
    #[serde(default = "default_grasp_step_size")]
    pub step_size: f64,
    #[serde(default = "default_grasp_perturbation_prob")]
    pub perturbation_prob: f64,
}

fn default_grasp_num_candidates() -> usize { 100 }
fn default_grasp_alpha() -> f64 { 0.3 }
fn default_grasp_num_neighbors() -> usize { 50 }
fn default_grasp_step_size() -> f64 { 0.1 }
fn default_grasp_perturbation_prob() -> f64 { 0.3 }