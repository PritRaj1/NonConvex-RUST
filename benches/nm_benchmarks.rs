mod common;
use common::fcns::{KBF, KBFConstraints};
use nalgebra::{SVector, SMatrix};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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
            let init_simplex = SMatrix::<f64, 3, 2>::from_columns(&[
                SVector::<f64, 3>::from_vec(vec![1.8, 1.0, 0.0]),
                SVector::<f64, 3>::from_vec(vec![0.5, 4.0, 0.0]),
            ]);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_simplex),
                KBF,
                None::<KBFConstraints>,
            );
            let _st = opt.run();
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
            let init_simplex = SMatrix::<f64, 3, 2>::from_columns(&[
                SVector::<f64, 3>::from_vec(vec![1.8, 1.0, 0.0]),
                SVector::<f64, 3>::from_vec(vec![0.5, 4.0, 0.0]),
            ]);
            let mut opt = NonConvexOpt::new(
                config.clone(),
                black_box(init_simplex),
                KBF,
                Some(KBFConstraints),
            );
            let _st = opt.run();
        })
    });
}

criterion_group!(benches, bench_nm_unconstrained, bench_nm_constrained);
criterion_main!(benches); 