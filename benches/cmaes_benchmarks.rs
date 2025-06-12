use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{SMatrix, SVector};
use serde_json;
use std::sync::LazyLock;

use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::Config;

mod common;
use common::fcns::{KBF, KBFConstraints};

static CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "CMAES": {
            "initial_sigma": 1.0,
        }
    }
}"#;

static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    serde_json::from_str(CONFIG_JSON).unwrap()
});

fn bench_cmaes_unconstrained(c: &mut Criterion) {
    c.bench_function("cmaes_unconstrained", |b| {
        b.iter(|| {
            let init_x = SVector::<f64, 2>::from_fn(|_,_| rand::random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(SMatrix::<f64, 100, 2>::from_vec(init_x.iter().cloned().collect())),
                KBF,
                None::<KBFConstraints>,
            );
            let _st = opt.run();
        })
    });
}

fn bench_cmaes_constrained(c: &mut Criterion) {
    c.bench_function("cmaes_constrained", |b| {
        b.iter(|| {
            let init_x = SVector::<f64, 2>::from_fn(|_,_| rand::random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(SMatrix::<f64, 1, 2>::from_vec(init_x.iter().cloned().collect())),
                KBF,
                Some(KBFConstraints)
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_cmaes_unconstrained, bench_cmaes_constrained);
criterion_main!(benches); 