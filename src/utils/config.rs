use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error; 
use serde_with::serde_as;
use serde_with::DisplayFromStr;

pub use crate::utils::alg_conf::{
    cga_conf::{CGAConf, CommonConf, CrossoverConf, SelectionConf}
};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum AlgConf {
    CGA(CGAConf),
    PT(PTConf),
    TS(TabuConf),
    Adam(AdamConf),
    GRASP(GRASPConf),
    SGA(SGAConf),
    NM(NelderMeadConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Config {
    pub opt_conf: OptConf,
    pub alg_conf: AlgConf,
}

#[serde_as]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OptConf {
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_atol")]
    pub atol: f64,
    #[serde(default = "default_rtol_max_iter_fraction")] 
    pub rtol_max_iter_fraction: f64,
}

fn default_max_iter() -> usize { 1000 }
fn default_rtol() -> f64 { 1e-6 }
fn default_atol() -> f64 { 1e-6 }
fn default_rtol_max_iter_fraction() -> f64 { 1.0 }

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PTConf {
    #[serde(default = "default_num_replicas")]
    pub num_replicas: usize,
    #[serde(default = "default_power_law_init")]
    pub power_law_init: f64,
    #[serde(default = "default_power_law_final")]
    pub power_law_final: f64,
    #[serde(default = "default_power_law_cycles")]
    pub power_law_cycles: usize,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_omega")]
    pub omega: f64,
    #[serde(default = "default_swap_check_type")]
    pub swap_check_type: String,
    #[serde(default = "default_swap_frequency")]
    pub swap_frequency: f64,
    #[serde(default = "default_swap_probability")]
    pub swap_probability: f64,
    #[serde(default = "default_mala_step_size")]
    pub mala_step_size: f64,
}

fn default_num_replicas() -> usize { 10 }
fn default_power_law_init() -> f64 { 2.0 }
fn default_power_law_final() -> f64 { 0.5 }
fn default_power_law_cycles() -> usize { 1 }
fn default_alpha() -> f64 { 0.1 }
fn default_omega() -> f64 { 2.1 }
fn default_swap_check_type() -> String { "Always".to_string() }
fn default_swap_frequency() -> f64 { 1.0 }
fn default_swap_probability() -> f64 { 0.1 }
fn default_mala_step_size() -> f64 { 0.01 }

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TabuConf {
    #[serde(default = "default_tabu_list_size")]
    pub tabu_list_size: usize,
    #[serde(default = "default_num_neighbors")]
    pub num_neighbors: usize,
    #[serde(default = "default_step_size")]
    pub step_size: f64,
    #[serde(default = "default_perturbation_prob")]
    pub perturbation_prob: f64,
    #[serde(default = "default_tabu_threshold")] // Aspiration
    pub tabu_threshold: f64,
    #[serde(default = "default_tabu_type")]
    pub tabu_type: String,
    // Reactive parameters
    #[serde(default = "default_min_tabu_size")]
    pub min_tabu_size: usize,
    #[serde(default = "default_max_tabu_size")]
    pub max_tabu_size: usize,
    #[serde(default = "default_increase_factor")]
    pub increase_factor: f64,
    #[serde(default = "default_decrease_factor")]
    pub decrease_factor: f64,
}

fn default_tabu_list_size() -> usize { 20 }
fn default_num_neighbors() -> usize { 50 }
fn default_step_size() -> f64 { 0.1 }
fn default_perturbation_prob() -> f64 { 0.3 }
fn default_tabu_threshold() -> f64 { 1e-6 }
fn default_tabu_type() -> String { "Standard".to_string() }
fn default_min_tabu_size() -> usize { 10 }
fn default_max_tabu_size() -> usize { 30 }
fn default_increase_factor() -> f64 { 1.1 }
fn default_decrease_factor() -> f64 { 0.9 }

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AdamConf {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,      
    #[serde(default = "default_beta1")]
    pub beta1: f64,    
    #[serde(default = "default_beta2")]
    pub beta2: f64,     
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,    
}

fn default_learning_rate() -> f64 { 0.001 }
fn default_beta1() -> f64 { 0.9 }
fn default_beta2() -> f64 { 0.999 }
fn default_epsilon() -> f64 { 1e-8 }

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

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SGAConf {
    #[serde(default = "default_sga_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_sga_momentum")]
    pub momentum: f64,
}

fn default_sga_learning_rate() -> f64 { 0.01 }
fn default_sga_momentum() -> f64 { 0.9 }

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NelderMeadConf {
    #[serde(default = "default_nm_alpha")]
    pub alpha: f64,  // Reflection coefficient
    #[serde(default = "default_nm_gamma")]
    pub gamma: f64,  // Expansion coefficient
    #[serde(default = "default_nm_rho")]
    pub rho: f64,    // Contraction coefficient
    #[serde(default = "default_nm_sigma")]
    pub sigma: f64,  // Shrink coefficient
}

fn default_nm_alpha() -> f64 { 1.0 }
fn default_nm_gamma() -> f64 { 2.0 }
fn default_nm_rho() -> f64 { 0.5 }
fn default_nm_sigma() -> f64 { 0.5 }

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to deserialize configuration: {0}")]
    DeserializationError(String),

    #[error("Failed to serialize configuration: {0}")]
    SerializationError(String),
}

// Have the option to load config from a json
impl Config {
    // Deserialize the json to config
    pub fn new(config: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(config)
            .map_err(|e| ConfigError::DeserializationError(e.to_string()))
    }

    // Serialize the config to json
    pub fn to_json(&self) -> Result<String, ConfigError> {
        serde_json::to_string(self)
            .map_err(|e| ConfigError::SerializationError(e.to_string()))
    }

    #[cfg(test)]
    pub fn from_str(json: &str) -> Self {
        serde_json::from_str(json).expect("Failed to parse config JSON")
    }
}
