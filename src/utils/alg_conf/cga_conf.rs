use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CGAConf {    
    pub common: CommonConf,
    pub crossover: CrossoverConf,
    pub selection: SelectionConf,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CommonConf {
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_num_parents")]
    pub num_parents: usize,
}

fn default_population_size() -> usize { 100 }
fn default_num_parents() -> usize { 2 }

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum CrossoverConf {
    Random(RandomCrossoverConf),
    Heuristic(HeuristicCrossoverConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct RandomCrossoverConf {
    #[serde(default = "default_crossover_prob")]
    pub crossover_prob: f64,
}

fn default_crossover_prob() -> f64 { 0.8 }

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct HeuristicCrossoverConf {
    #[serde(default = "default_crossover_prob")]
    pub crossover_prob: f64,
}

fn default_crossover_prob() -> f64 { 0.8 }
#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum SelectionConf {
    RouletteWheel(RouletteWheelSelectionConf),
    Tournament(TournamentSelectionConf),
    Residual(ResidualSelectionConf),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct RouletteWheelSelectionConf {}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TournamentSelectionConf {
    #[serde(default = "default_tournament_size")]
    pub tournament_size: usize,
}

fn default_tournament_size() -> usize { 5 }

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ResidualSelectionConf {}