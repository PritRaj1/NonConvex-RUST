use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::Config;
use rand::random;
use std::sync::LazyLock;

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
        "DE": {
            "population_size": 50,
            "f": 0.8,
            "cr": 0.9,
            "strategy": "Rand1Bin"
        }
    }
}"#;

static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    serde_json::from_str(CONFIG_JSON).unwrap()
});

fn bench_de_unconstrained(c: &mut Criterion) {
    c.bench_function("de_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(50, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                KBF,
                None::<KBFConstraints>
            );
            opt.run()
        })
    });
}

fn bench_de_constrained(c: &mut Criterion) {
    c.bench_function("de_constrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(50, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                CONFIG.clone(),
                black_box(init_pop),
                KBF,
                Some(KBFConstraints)
            );
            opt.run()
        })
    });
}

criterion_group!(benches, bench_de_unconstrained, bench_de_constrained);
criterion_main!(benches); 