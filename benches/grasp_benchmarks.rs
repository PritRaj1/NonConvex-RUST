use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use rand::random;

use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, GRASPConf};

mod common;
use common::fcns::{KBF, KBFConstraints};

fn bench_grasp_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::GRASP(GRASPConf {
            num_candidates: 50,
            alpha: 0.3,
            num_neighbors: 20,
            step_size: 0.1,
            perturbation_prob: 0.3,
        }),
    };

    c.bench_function("grasp_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                KBF,
                None::<KBFConstraints>
            );
            let _st = opt.run();
        })
    });
}

fn bench_grasp_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::GRASP(GRASPConf {
            num_candidates: 50,
            alpha: 0.3,
            num_neighbors: 20,
            step_size: 0.1,
            perturbation_prob: 0.3,
        }),
    };

    c.bench_function("grasp_constrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                KBF,
                Some(KBFConstraints)
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_grasp_unconstrained, bench_grasp_constrained);
criterion_main!(benches); 