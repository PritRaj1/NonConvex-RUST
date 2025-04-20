use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error; 
use serde_with::serde_as;
use serde_with::DisplayFromStr;

pub use crate::utils::alg_conf::{
    cga_conf::{CGAConf, CommonConf, CrossoverConf, SelectionConf},
    pt_conf::{PTConf, SwapConf},
    tabu_conf::{TabuConf, ListType, ReactiveConf, StandardConf},
    adam_conf::AdamConf,
    grasp_conf::GRASPConf,
    sga_conf::SGAConf,
    nm_conf::NelderMeadConf,
    lbfgs_conf::{LBFGSConf, LineSearchConf, BacktrackingConf, StrongWolfeConf, HagerZhangConf, MoreThuenteConf, GoldenSectionConf},
    mspo_conf::MSPOConf,
    sa_conf::SAConf,
    de_conf::{DEConf, DEStrategy},
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
    LBFGS(LBFGSConf),
    MSPO(MSPOConf),
    SA(SAConf),
    DE(DEConf),
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
