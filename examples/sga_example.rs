mod common;

use nalgebra::SMatrix;
use plotters::prelude::*;
use gif::Frame;
use image::ImageReader;

use common::fcns::{MultiModalFunction, BoxConstraints};
use common::img::{create_contour_data, setup_gif, find_closest_color, setup_chart, get_color_palette};

use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, SGAConf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 100,
            rtol: 1e-6,
            atol: 1e-6,
            rtol_max_iter_fraction: 1.0,
        },
        alg_conf: AlgConf::SGA(SGAConf {
            learning_rate: 0.05,
            momentum: 0.9,
        }),
    };

    let obj_f = MultiModalFunction;
    let constraints = BoxConstraints;

    let mut opt = NonConvexOpt::new(
        config, 
        SMatrix::<f64, 1, 2>::from_vec(vec![4.0, 9.0]), 
        obj_f.clone(), 
        Some(constraints.clone())
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/sga_kbf.gif")?;

    for frame in 0..100 {
        let mut chart = setup_chart(
            frame,
            "Stochastic Gradient Ascent",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/sga_frame.png",
        )?;

        // Draw current individual
        let population = opt.get_population();
        let current_x = population.column(0);
        chart.draw_series(std::iter::once(Circle::new(
            (current_x[0], current_x[1]),
            6,
            RGBColor(255, 0, 0).filled(),
        )))?;

        // Draw best individual
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        chart.plotting_area().present()?;
        
        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/sga_frame.png")?
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
        frame.delay = 4;
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/sga_frame.png")?;

    Ok(())
} 