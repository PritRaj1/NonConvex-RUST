mod common;
use common::fcns::{KBF, KBFConstraints};
use common::img::{create_contour_data, setup_gif, find_closest_color, setup_chart, get_color_palette};
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, GRASPConf};
use nalgebra::SMatrix;
use plotters::prelude::*;
use gif::Frame;
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        opt_conf: OptConf {
            max_iter: 50,
            rtol: 1e-6,
            atol: 1e-6,
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

    let obj_f = KBF;
    let constraints = KBFConstraints;

    let mut opt = NonConvexOpt::new(
        config, 
        SMatrix::<f64, 1, 2>::from_vec(vec![
            rand::random::<f64>() * 10.0,
            rand::random::<f64>() * 10.0
        ]),
        obj_f.clone(), 
        Some(constraints.clone())
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/grasp_kbf.gif")?;

    for frame in 0..50 {
        let mut chart = setup_chart(
            frame,
            "GRASP",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/grasp_frame.png",
        )?;

        // Draw current solution in red
        let population = opt.get_population();
        let current_x = population.column(0);
        chart.draw_series(std::iter::once(Circle::new(
            (current_x[0], current_x[1]),
            6,
            RGBColor(255, 0, 0).filled(),
        )))?;

        // Draw best solution in yellow
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 255, 0).filled(),
        )))?;

        chart.plotting_area().present()?;
        
        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/grasp_frame.png")?
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
        frame.delay = 10; 
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/grasp_frame.png")?;   

    Ok(())
} 