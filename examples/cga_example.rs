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
            "max_iter": 100,
            "rtol": "1e-6",
            "atol": "1e-6",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "CGA": {
                "common": {
                    "num_parents": 40
                },
                "crossover": {
                    "Heuristic": {
                        "crossover_prob": 0.5
                    }
                },
                "selection": {
                    "Residual": {}
                },
                "mutation": {
                    "NonUniform": {
                        "mutation_rate": 0.1,
                        "b": 5.0
                    }
                }
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json).unwrap();

    let obj_f = KBF;
    let constraints = KBFConstraints;

    // Initialize population with random points
    let mut init_pop = SMatrix::<f64, 100, 2>::zeros();
    for i in 0..100 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 10.0;
        }
    }

    let mut opt = NonConvexOpt::new(config, init_pop, obj_f.clone(), Some(constraints.clone()));

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/cga_kbf.gif")?;

    for frame in 0..10 {
        let mut chart = setup_chart(
            frame,
            "CGA",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/cga_frame.png",
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
        let img = ImageReader::open("examples/cga_frame.png")?
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
        frame.delay = 40; 
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/cga_frame.png")?;

    Ok(())
}