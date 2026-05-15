use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::LazyLock;

use non_convex_opt::utils::config::Config;

mod common;
use common::benchmark_utils::{benchmark_optimization, BenchmarkConfig};

static TPE_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 50,
        "rtol": 0.0,
        "atol": 0.0
    },
    "alg_conf": {
        "TPE": {
            "n_initial_random": 20,
            "n_candidates": 100,
            "gamma": 0.25,
            "max_history": 1000
        }
    }
}"#;

static TPE_CONFIG: LazyLock<Config> =
    LazyLock::new(|| serde_json::from_str(TPE_CONFIG_JSON).unwrap());

fn bench_tpe(c: &mut Criterion) {
    let bench_config = BenchmarkConfig::default();

    c.bench_function("tpe", |b| {
        b.iter(|| {
            benchmark_optimization(&TPE_CONFIG, &bench_config);
        })
    });
}

criterion_group!(benches, bench_tpe);
criterion_main!(benches);
