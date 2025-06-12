mod common;

use serde_json;
use nalgebra::SMatrix;
use plotters::prelude::*;
use gif::Frame;
use image::ImageReader;

use common::fcns::{KBF, KBFConstraints};
use common::img::{create_contour_data, setup_gif, find_closest_color, setup_chart, get_color_palette};

use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let config_json = r#"
    {
        "opt_conf": {
            "max_iter": 20,
            "rtol": "1e-6",
            "atol": "1e-6",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "CMAES": {
                "num_parents": 50,
                "initial_sigma": 1.5
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json).unwrap();

    let obj_f = KBF;
    let constraints = KBFConstraints;

    let mut opt = NonConvexOpt::new(
        config,
        SMatrix::<f64, 20, 2>::from_vec({
            let mut v = vec![4.0, 9.0];
            v.resize(20 * 2, 0.0);
            v
        }),
        obj_f.clone(), 
        Some(constraints.clone())
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/cmaes_kbf.gif")?;

    for frame in 0..20 {
        let mut chart = setup_chart(
            frame,
            "CMAES",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/cmaes_frame.png",
        )?;

        // Draw population
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
        let img = ImageReader::open("examples/cmaes_frame.png")?
            .decode()?
            .into_rgb8();
        
        let mut indexed_pixels = Vec::with_capacity((img.width() * img.height()) as usize);
        for pixel in img.pixels() {
            let idx = find_closest_color(pixel[0], pixel[1], pixel[2], &color_palette);
            indexed_pixels.push(idx as u8);
        }
        
        let mut frame = Frame::default();
        frame.width = 800;
        frame.height = 800;
        frame.delay = 20; 
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/cmaes_frame.png")?;

    Ok(())
}