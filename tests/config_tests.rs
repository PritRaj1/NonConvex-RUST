use non_convex_opt::utils::config::{Config, AlgConf, CGAConf, OptConf};
use serde_json;

#[test]
fn test_deserialize_config() {
    let json = r#"
    {
        "opt_conf": {
            "max_iter": 500,
            "rtol": 1e-5,
            "atol": 1e-4
        },
        "alg_conf": {
            "CGA": {
                "pop_size": 200,
                "num_parents": 4,
                "selection_method": "roulette",
                "mating_method": "blend",
                "crossover_prob": 0.9,
            }
        }
    }
    "#;

    let config: Config = serde_json::from_str(json).unwrap();
    assert_eq!(config.opt_conf.max_iter, 500);
    if let AlgConf::CGA(cga_conf) = &config.alg_conf {
        assert_eq!(cga_conf.pop_size, 200);
    } else {
        panic!("Expected CGAConf, but got something else");
    }
}

fn test_serialize_config() {
    let config = Config {
        opt_conf: OptConf { max_iter: 500, rtol: 1e-5, atol: 1e-4 },
        alg_conf: AlgConf::CGA(CGAConf { pop_size: 200, num_parents: 4, selection_method: "roulette".to_string(), mating_method: "blend".to_string(), crossover_prob: 0.9, tournament_size: 2 }),
    };
    let serialized = serde_json::to_string(&config).unwrap();
    assert_eq!(serialized, r#"{"opt_conf":{"max_iter":500,"rtol":1e-5,"atol":1e-4},"alg_conf":{"CGA":{"pop_size":200,"num_parents":4,"selection_method":"roulette","mating_method":"blend","crossover_prob":0.9,"tournament_size":2}}}"#);
}
