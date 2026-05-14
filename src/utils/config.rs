use serde::{Deserialize, Serialize};

pub use crate::utils::alg_conf::{
    adam_conf::AdamConf,
    cem_conf::CEMConf,
    cga_conf::{CGAConf, CommonConf, CrossoverConf, MutationConf, SelectionConf},
    cmaes_conf::CMAESConf,
    de_conf::{DEConf, DEStrategy},
    grasp_conf::GRASPConf,
    lbfgs_conf::{
        BacktrackingConf, GoldenSectionConf, HagerZhangConf, LBFGSConf, LineSearchConf,
        MoreThuenteConf, StrongWolfeConf,
    },
    mspo_conf::MSPOConf,
    nm_conf::NelderMeadConf,
    pt_conf::{PTConf, SwapConf},
    sa_conf::SAConf,
    sga_conf::SGAConf,
    tabu_conf::{ListType, ReactiveConf, StandardConf, TabuConf},
    tpe_conf::TPEConf,
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
    CMAES(CMAESConf),
    TPE(TPEConf),
    CEM(CEMConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Config {
    pub opt_conf: OptConf,
    pub alg_conf: AlgConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct OptConf {
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    #[serde(default = "default_atol")]
    pub atol: f64,
    #[serde(default = "default_rtol_max_iter_fraction")]
    pub rtol_max_iter_fraction: f64,
    #[serde(default = "default_stagnation_window")]
    pub stagnation_window: usize,
}

fn default_max_iter() -> usize {
    1000
}
fn default_rtol() -> f64 {
    1e-6
}
fn default_atol() -> f64 {
    1e-6
}
fn default_rtol_max_iter_fraction() -> f64 {
    1.0
}
fn default_stagnation_window() -> usize {
    50
}

#[derive(Debug)]
pub enum ConfigError {
    Deserialization(serde_json::Error),
    Serialization(serde_json::Error),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Deserialization(e) => write!(f, "failed to deserialize config: {e}"),
            Self::Serialization(e) => write!(f, "failed to serialize config: {e}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Deserialization(e) | Self::Serialization(e) => Some(e),
        }
    }
}

impl Config {
    pub fn new(config: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(config).map_err(ConfigError::Deserialization)
    }

    pub fn to_json(&self) -> Result<String, ConfigError> {
        serde_json::to_string(self).map_err(ConfigError::Serialization)
    }

    #[cfg(test)]
    pub fn from_json_str(json: &str) -> Self {
        serde_json::from_str(json).expect("Failed to parse config JSON")
    }
}
