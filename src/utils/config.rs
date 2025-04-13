use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error; 
use serde_with::serde_as;
use serde_with::DisplayFromStr;

#[derive(Deserialize, Serialize, Debug)]
pub enum AlgConf {
    CGA(CGAConf),
    PT(PTConf),
    TS(TabuConf),
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Config {
    pub opt_conf: OptConf,
    pub alg_conf: AlgConf,
}

#[serde_as]
#[derive(Deserialize, Serialize, Debug)]
pub struct OptConf {
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    #[serde_as(as = "DisplayFromStr")]
    #[serde(default = "default_atol")]
    pub atol: f64,
}

fn default_max_iter() -> usize { 1000 }
fn default_rtol() -> f64 { 1e-6 }
fn default_atol() -> f64 { 1e-6 }

#[derive(Deserialize, Serialize, Debug)]
pub struct CGAConf {    
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_num_parents")]
    pub num_parents: usize,
    #[serde(default = "default_selection_method")]
    pub selection_method: String,
    #[serde(default = "default_crossover_method")]
    pub crossover_method: String,
    #[serde(default = "default_crossover_prob")]
    pub crossover_prob: f64,
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,
}

fn default_population_size() -> usize { 100 }
fn default_num_parents() -> usize { 2 }
fn default_selection_method() -> String { "tournament".to_string() }
fn default_crossover_method() -> String { "uniform".to_string() }
fn default_crossover_prob() -> f64 { 0.8 }
fn default_tournament_size() -> usize { 2 }

#[derive(Deserialize, Serialize, Debug)]
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

#[derive(Deserialize, Serialize, Debug)]
pub struct TabuConf {
    #[serde(default = "default_tabu_list_size")]
    pub tabu_list_size: usize,
    #[serde(default = "default_num_neighbors")]
    pub num_neighbors: usize,
    #[serde(default = "default_step_size")]
    pub step_size: f64,
    #[serde(default = "default_perturbation_prob")]
    pub perturbation_prob: f64,
    #[serde(default = "default_tabu_threshold")]
    pub tabu_threshold: f64,
}

fn default_tabu_list_size() -> usize { 20 }
fn default_num_neighbors() -> usize { 50 }
fn default_step_size() -> f64 { 0.1 }
fn default_perturbation_prob() -> f64 { 0.3 }
fn default_tabu_threshold() -> f64 { 1e-6 }

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
