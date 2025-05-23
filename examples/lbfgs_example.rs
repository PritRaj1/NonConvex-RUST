mod common;
use common::fcns::{MultiModalFunction, BoxConstraints};
use common::img::{create_contour_data, setup_gif, find_closest_color, setup_chart, get_color_palette};
use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::Config;
use nalgebra::DMatrix;
use plotters::prelude::*;
use gif::Frame;
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let config_json = r#"
    {
        "opt_conf": {
            "max_iter": 10,
            "rtol": "1e-6",
            "atol": "1e-6",
            "rtol_max_iter_fraction": 1.0
        },
        "alg_conf": {
            "LBFGS": {
                "common": {
                    "memory_size": 10
                },
                "line_search": {
                    "Backtracking": {
                        "c1": 0.0001,
                        "c2": 0.9,
                        "max_iters": 100
                    }
                }
            }
        }
    }"#;

    let config: Config = serde_json::from_str(config_json)?;

    let obj_f = MultiModalFunction;
    let constraints = BoxConstraints;

    let mut opt = NonConvexOpt::new(
        config, 
        DMatrix::from_row_slice(1, 2, &[4.0, 9.0]),
        obj_f.clone(), 
        Some(constraints.clone())
    );

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);
    let color_palette = get_color_palette();
    let mut encoder = setup_gif("examples/gifs/lbfgs_kbf.gif")?;

    for frame in 0..10 {
        let mut chart = setup_chart(
            frame,
            "LBFGS",
            resolution,
            &z_values,
            min_val,
            max_val,
            &constraints,
            "examples/lbfgs_frame.png",
        )?;

        // Draw best individual in yellow
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(Circle::new(
            (best_x[0], best_x[1]),
            6,
            RGBColor(255, 0, 0).filled(),
        )))?;

        // Save frame and convert to GIF
        chart.plotting_area().present()?;
        
        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/lbfgs_frame.png")?
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
        frame.delay = 40; 
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/lbfgs_frame.png")?;   

    Ok(())
}
