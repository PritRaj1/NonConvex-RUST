use serde::Deserialize
use serde_json;
use std::error::Error;

#[derive(Deserialize, Debug)]
pub enum AlgConf {
    CGA(CGAConf),
    PT(PTConf),
}

#[derive(Deserialize, Debug)]
pub struct Config {
    pub opt_conf: OptConf,
    pub alg_conf: AlgConf,
}

#[derive(Debug, Deserialize)]
pub struct OptConf {
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    #[serde(default = "default_atol")]
    pub atol: f64,
}

fn default_max_iter() -> usize { 1000 }
fn default_rtol() -> f64 { 1e-6 }
fn default_atol() -> f64 { 1e-6 }

#[derive(Debug, Deserialize)]
pub struct CGAConf {    
    #[serde(default = "default_pop_size")]
    pub pop_size: usize,
    #[serde(default = "default_num_parents")]
    pub num_parents: usize,
    #[serde(default = "default_selection_method")]
    pub selection_method: String,
    #[serde(default = "default_mating_method")]
    pub mating_method: String,
    #[serde(default = "default_crossover_prob")]
    pub crossover_prob: f64,
}

fn default_pop_size() -> usize { 100 }
fn default_num_parents() -> usize { 2 }
fn default_selection_method() -> String { "tournament".to_string() }
fn default_mating_method() -> String { "uniform".to_string() }
fn default_crossover_prob() -> f64 { 0.8 }

#[derive(Debug, Deserialize)]
pub struct PTConf {
    #[serde(default = "default_num_replicas")]
    pub num_replicas: usize,
    #[serde(default = "default_num_chains")]
    pub num_chains: usize,
    #[serde(default = "default_power_law")]
    pub power_law: f64,
    #[serde(default = "default_exchange_type")]
    pub exchange_type: String,
}

fn default_num_replicas() -> usize { 10 }
fn default_num_chains() -> usize { 10 }
fn default_power_law() -> f64 { 0.5 }
fn default_exchange_type() -> String { "swap".to_string() }

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
}
