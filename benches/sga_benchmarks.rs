use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::random;
use nalgebra::DMatrix;

use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, SGAConf};

mod common;
use common::fcns::{RosenbrockFunction, RosenbrockConstraints};

fn bench_sga_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::SGA(SGAConf {
            learning_rate: 0.01,
            momentum: 0.9,
        }),
    };

    c.bench_function("sga_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                RosenbrockFunction,
                None::<RosenbrockConstraints>
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_sga_unconstrained);
criterion_main!(benches); 