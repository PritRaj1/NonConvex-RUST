use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config};
use rand::random;
use std::sync::LazyLock;
use serde_json;

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
        "TS": {
            "common": {
                "num_neighbors": 100,
                "step_size": 1.5,
                "perturbation_prob": 0.3,
                "tabu_list_size": 50,
                "tabu_threshold": 0.05,
            },
            "list_type": {
                "Standard": {}
            }
        }
    }
}
"#;

static REACTIVE_CONFIG_JSON: &str = r#"
{
    "opt_conf": {
        "max_iter": 10,
        "rtol": "1e-8",
        "atol": "1e-8",
        "rtol_max_iter_fraction": 1.0
    },
    "alg_conf": {
        "TS": {
            "common": {
                "num_neighbors": 100,
                "step_size": 1.5,
                "perturbation_prob": 0.3,
                "tabu_list_size": 50,
                "tabu_threshold": 0.05,
            },
            "list_type": {
                "Reactive": {
                    "min_tabu_size": 10,
                    "max_tabu_size": 30,
                    "increase_factor": 1.1,
                    "decrease_factor": 0.9
                }
            }
        }
    }
}
"#;

static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    serde_json::from_str(CONFIG_JSON).unwrap()
});

static REACTIVE_CONFIG: LazyLock<Config> = LazyLock::new(|| {
    serde_json::from_str(REACTIVE_CONFIG_JSON).unwrap()
});

fn bench_tabu_unconstrained(c: &mut Criterion) {

    c.bench_function("tabu_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
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

fn bench_tabu_constrained(c: &mut Criterion) {

    c.bench_function("tabu_constrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
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

fn bench_reactive_tabu_unconstrained(c: &mut Criterion) {

    c.bench_function("reactive_tabu_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                REACTIVE_CONFIG.clone(),
                black_box(init_pop),
                KBF,
                None::<KBFConstraints>
            );
            opt.run()
        })
    });
}

fn bench_reactive_tabu_constrained(c: &mut Criterion) {

    c.bench_function("reactive_tabu_constrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                REACTIVE_CONFIG.clone(),
                black_box(init_pop),
                KBF,
                Some(KBFConstraints)
            );
            opt.run()
        })
    });
}

criterion_group!(
    benches, 
    bench_tabu_unconstrained, 
    bench_tabu_constrained, 
    bench_reactive_tabu_unconstrained, 
    bench_reactive_tabu_constrained
);
criterion_main!(benches); 