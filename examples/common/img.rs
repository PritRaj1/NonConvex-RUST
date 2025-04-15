use nalgebra::DVector;
use plotters::prelude::*;
use plotters::coord::types::RangedCoordf64;
use gif::{Encoder, Repeat};
use std::fs::File;
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};

pub fn create_contour_data<F: ObjectiveFunction<f64>>(
    obj_f: &F, 
    resolution: usize
) -> (Vec<Vec<f64>>, f64, f64) {
    let mut z = vec![vec![0.0; resolution]; resolution];
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for i in 0..resolution {
        for j in 0..resolution {
            let x = 10.0 * i as f64 / (resolution - 1) as f64;
            let y = 10.0 * j as f64 / (resolution - 1) as f64;
            let point = DVector::from_vec(vec![x, y]);
            let val = obj_f.f(&point);
            z[i][j] = val;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }
    (z, min_val, max_val)
}

pub fn setup_gif(filename: &str) -> Result<Encoder<File>, Box<dyn std::error::Error>> {
    let gif = File::create(filename)?;
    let color_palette = get_color_palette();
    let mut encoder = Encoder::new(gif, 800, 800, &color_palette)?;
    encoder.set_repeat(Repeat::Infinite)?;
    Ok(encoder)
}

pub fn find_closest_color(r: u8, g: u8, b: u8, palette: &[u8]) -> usize {
    let mut best_diff = f64::INFINITY;
    let mut best_idx = 0;

    for i in (0..palette.len()).step_by(3) {
        let pr = palette[i];
        let pg = palette[i + 1];
        let pb = palette[i + 2];

        let diff = (
            (r as f64 - pr as f64).powi(2) +
            (g as f64 - pg as f64).powi(2) +
            (b as f64 - pb as f64).powi(2)
        ).sqrt();

        if diff < best_diff {
            best_diff = diff;
            best_idx = i / 3;
        }
    }
    best_idx
}

pub fn setup_chart<'a, F: BooleanConstraintFunction<f64>>(
    frame: usize,
    algorithm_name: &'a str,
    resolution: usize,
    z_values: &'a [Vec<f64>],
    min_val: f64,
    max_val: f64,
    constraints: &'a F,
    frame_path: &'a str,
) -> Result<ChartContext<'a, BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>, Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(frame_path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{}, Keane's Bump Function - Iteration {}", algorithm_name, frame),
            ("sans-serif", 30)
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..10f64, 0f64..10f64)?;

    chart.configure_mesh()
        .disable_mesh()    
        .draw()?;

    // Draw contour and feasible regions
    for i in 0..resolution-1 {
        for j in 0..resolution-1 {
            let x = 10.0 * i as f64 / (resolution - 1) as f64;
            let y = 10.0 * j as f64 / (resolution - 1) as f64;
            let dx = 10.0 / (resolution - 1) as f64; 
            let val = (z_values[i][j] - min_val) / (max_val - min_val);
            let color = RGBColor(
                (255.0 * val) as u8,
                (255.0 * val) as u8,
                (255.0 * val) as u8,
            );

            let point = DVector::from_vec(vec![x, y]);
            if !constraints.g(&point) {
                let stripe_width = 0.2;
                let stripe_pos = ((x + y) / stripe_width).floor() as i32;
                if stripe_pos % 2 == 0 {
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(x, y), (x + dx, y + dx)],  
                        RGBColor(128, 128, 128).mix(0.3).filled(),
                    )))?;
                }
            } else {
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x, y), (x + dx, y + dx)],
                    color.filled(),
                )))?;
            }
        }
    }

    Ok(chart)
}

pub fn get_color_palette() -> Vec<u8> {
    let mut color_palette = Vec::with_capacity(768);
    
    color_palette.extend_from_slice(&[
        255, 0, 0,      // Bright red for current individual
        255, 255, 0,    // Bright yellow for best individual
    ]);
    
    // Add grayscale colors
    for i in 0..254 {  
        color_palette.push(i as u8);    
        color_palette.push(i as u8);    
        color_palette.push(i as u8);   
    }
    
    color_palette
} 