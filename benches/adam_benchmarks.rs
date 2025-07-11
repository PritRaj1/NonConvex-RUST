use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::SMatrix;
use rand::random;

use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, AdamConf};

mod common;
use common::fcns::{RosenbrockFunction, RosenbrockConstraints};

fn bench_adam_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::Adam(AdamConf {
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }),
    };

    c.bench_function("adam_unconstrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>,
            );
            let _st = opt.run();
        })
    });
}

fn bench_adam_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::Adam(AdamConf {
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }),
    };

    c.bench_function("adam_constrained", |b| {
        b.iter(|| {
            let init_pop = SMatrix::<f64, 1, 2>::from_fn(|_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                Some(RosenbrockConstraints)
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_adam_unconstrained, bench_adam_constrained);
criterion_main!(benches); 