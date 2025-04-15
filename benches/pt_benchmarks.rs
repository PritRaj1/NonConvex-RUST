use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, PTConf};
use rand::random;

mod common;
use common::fcns::{KBF, KBFConstraints};

fn bench_pt_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
        },
        alg_conf: AlgConf::PT(PTConf {
            num_replicas: 10,
            power_law_init: 2.0,
            power_law_final: 0.5,
            power_law_cycles: 1,
            swap_check_type: "Always".to_string(),
            alpha: 0.1,
            omega: 2.1,
            swap_frequency: 0.9,
            swap_probability: 0.1,
            mala_step_size: 0.1,
        }),
    };

    c.bench_function("pt_unconstrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(10, 2, |_, _| random::<f64>() * 10.0);
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

fn bench_pt_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
        },
        alg_conf: AlgConf::PT(PTConf {
            num_replicas: 10,
            power_law_init: 2.0,
            power_law_final: 0.5,
            power_law_cycles: 1,
            swap_check_type: "Always".to_string(),
            alpha: 0.1,
            omega: 2.1,
            swap_frequency: 0.9,
            swap_probability: 0.1,
            mala_step_size: 0.1,
        }),
    };

    c.bench_function("pt_constrained", |b| {
        b.iter(|| {
            let init_pop = DMatrix::from_fn(10, 2, |_, _| random::<f64>() * 10.0);
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

criterion_group!(benches, bench_pt_unconstrained, bench_pt_constrained);
criterion_main!(benches); 