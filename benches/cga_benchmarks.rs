use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, CGAConf};
use rand::random;

mod common;
use common::fcns::{KBF, KBFConstraints};

fn bench_cga_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
        },
        alg_conf: AlgConf::CGA(CGAConf {
            population_size: 100,
            num_parents: 20,
            selection_method: "Tournament".to_string(),
            crossover_method: "Heuristic".to_string(),
            crossover_prob: 0.5,
            tournament_size: 50,
        }),
    };

    c.bench_function("cga_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(100, 2, |_, _| random::<f64>() * 10.0);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_pop),
                KBF,
                None::<KBFConstraints>,
            );
            opt.run()
        })
    });
}

fn bench_cga_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
        },
        alg_conf: AlgConf::CGA(CGAConf {
            population_size: 100,
            num_parents: 20,
            selection_method: "Tournament".to_string(),
            crossover_method: "Heuristic".to_string(),
            crossover_prob: 0.5,
            tournament_size: 50,
        }),
    };

    c.bench_function("cga_constrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(100, 2, |_, _| random::<f64>() * 10.0);
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

criterion_group!(benches, bench_cga_unconstrained, bench_cga_constrained);
criterion_main!(benches); 