mod common;
use common::fcns::{KBF, KBFConstraints};
use common::img::{create_contour_data, setup_gif, find_closest_color, setup_chart, get_color_palette};
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, PTConf};
use nalgebra::DMatrix;
use plotters::prelude::*;
use gif::Frame;
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 1e-6,
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

    let obj_f = KBF;
    let constraints = KBFConstraints;

    // Initialize population with random points
    let mut init_pop = DMatrix::zeros(10, 2);
    for i in 0..10 {
        for j in 0..2 {
            init_pop[(i, j)] = 3.0 + (rand::random::<f64>() * 5.0);
        }
    }

    let mut opt = NonConvexOpt::new(config, init_pop, obj_f.clone(), Some(constraints.clone()));

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/pt_kbf.gif")?;

    for frame in 0..100 {
        let mut chart = setup_chart(
            frame,
            "Parallel Tempering",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/pt_frame.png",
        )?;

        // Draw all replicas
        let population = opt.get_population();
        chart.draw_series(
            population.row_iter().map(|row| {
                Circle::new(
                    (row[0], row[1]), 
                    3, 
                    RGBColor(255, 0, 0).filled()
                )
            })
        )?;

        // Draw best individual
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(
            Circle::new(
                (best_x[0], best_x[1]),
                6, 
                RGBColor(255, 255, 0).filled()
            )
        ))?;

        chart.plotting_area().present()?;
        
        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/pt_frame.png")?
            .decode()?
            .into_rgba8();
        
        let mut indexed_pixels = Vec::with_capacity((img.width() * img.height()) as usize);
        for pixel in img.pixels() {
            let idx = find_closest_color(pixel[0], pixel[1], pixel[2], &color_palette);
            indexed_pixels.push(idx as u8);
        }
        
        let mut frame = Frame::default();
        frame.width = 800;
        frame.height = 800;
        frame.delay = 5;
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/pt_frame.png")?;

    Ok(())
}