mod common;
use common::fcns::{KBF, KBFConstraints};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{DVector, DMatrix};
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, NelderMeadConf};

fn bench_nm_unconstrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::NM(NelderMeadConf {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }),
    };

    c.bench_function("nm_unconstrained", |b| {
        b.iter(|| {
            let init_simplex = DMatrix::from_columns(&[
                DVector::from_vec(vec![1.6, 0.55]),
                DVector::from_vec(vec![1.5, 0.6]),
                DVector::from_vec(vec![1.7, 0.5]),
            ]);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_simplex),
                KBF,
                None::<KBFConstraints>,
            );
            opt.run()
        })
    });
}

fn bench_nm_constrained(c: &mut Criterion) {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 10,
            rtol: -1e8,
            atol: -1e8,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::NM(NelderMeadConf {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }),
    };

    c.bench_function("nm_constrained", |b| {
        b.iter(|| {
            let init_simplex = DMatrix::from_columns(&[
                DVector::from_vec(vec![1.6, 0.55]),
                DVector::from_vec(vec![1.5, 0.6]),
                DVector::from_vec(vec![1.7, 0.5]),
            ]);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_simplex),
                KBF,
                Some(KBFConstraints),
            );
            opt.run()
        })
    });
}

criterion_group!(benches, bench_nm_unconstrained, bench_nm_constrained);
criterion_main!(benches); 