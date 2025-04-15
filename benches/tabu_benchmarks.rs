use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, TabuConf};
use rand::random;

mod common;
use common::fcns::{KBF, KBFConstraints};

fn bench_tabu_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 1e-6,
        },
        alg_conf: AlgConf::TS(TabuConf {
            num_neighbors: 100,
            step_size: 1.5,
            perturbation_prob: 0.3,
            tabu_list_size: 50,
            tabu_threshold: 0.05,
        }),
    };

    c.bench_function("tabu_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                KBF,
                None::<KBFConstraints>
            );
            opt.run()
        })
    });
}

fn bench_tabu_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 1e-6,
        },
        alg_conf: AlgConf::TS(TabuConf {
            num_neighbors: 100,
            step_size: 1.5,
            perturbation_prob: 0.3,
            tabu_list_size: 50,
            tabu_threshold: 0.05,
        }),
    };

    c.bench_function("tabu_constrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(1, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                KBF,
                Some(KBFConstraints)
            );
            opt.run()
        })
    });
}

criterion_group!(benches, bench_tabu_unconstrained, bench_tabu_constrained);
criterion_main!(benches); 