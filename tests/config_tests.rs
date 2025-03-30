use non_convex_opt::utils::config::{OptConf, CGAConf, AlgConf, Config};
use serde_json;

#[test]
fn test_deserialize_config() {
    let json = r#"{
        "opt_conf": {
            "max_iter": 500,
            "rtol": "1e-5",
            "atol": "1e-4"
        },
        "alg_conf": {
            "CGA": {
                "pop_size": 200,
                "num_parents": 4,
                "selection_method": "roulette",
                "mating_method": "Random",
                "crossover_prob": 0.9,
                "tournament_size": 2
            }
        }
    }"#;

    let config: Config = serde_json::from_str(json).unwrap();
    assert_eq!(config.opt_conf.max_iter, 500);
    assert_eq!(config.opt_conf.rtol, 1e-5);
    assert_eq!(config.opt_conf.atol, 1e-4);
}

#[test]
fn test_serialize_config() {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 500,
            rtol: 1e-5,
            atol: 1e-4,
        },
        alg_conf: AlgConf::CGA(CGAConf {
            pop_size: 200,
            num_parents: 4,
            selection_method: "roulette".to_string(),
            mating_method: "Random".to_string(),
            crossover_prob: 0.9,
            tournament_size: 2,
        }),
    };

    let serialized = serde_json::to_string(&config).unwrap();
    let deserialized: Config = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(deserialized.opt_conf.max_iter, config.opt_conf.max_iter);
    assert_eq!(deserialized.opt_conf.rtol, config.opt_conf.rtol);
    assert_eq!(deserialized.opt_conf.atol, config.opt_conf.atol);
    
    if let (AlgConf::CGA(orig), AlgConf::CGA(de)) = (&config.alg_conf, &deserialized.alg_conf) {
        assert_eq!(de.pop_size, orig.pop_size);
        assert_eq!(de.num_parents, orig.num_parents);
        assert_eq!(de.selection_method, orig.selection_method);
        assert_eq!(de.mating_method, orig.mating_method);
        assert_eq!(de.crossover_prob, orig.crossover_prob);
        assert_eq!(de.tournament_size, orig.tournament_size);
    } else {
        panic!("Expected CGAConf");
    }
}
