use non_convex_opt::NonConvexOpt;
use non_convex_opt::utils::config::{Config, OptConf, AlgConf, PTConf};
use non_convex_opt::utils::opt_prob::{ObjectiveFunction, BooleanConstraintFunction};
use nalgebra::{DVector, DMatrix};
use plotters::prelude::*;
use gif::{Frame, Encoder, Repeat};
use std::fs::File;
use image::ImageReader;

#[derive(Clone)]
struct KBF;

impl ObjectiveFunction<f64> for KBF {
    fn f(&self, x: &DVector<f64>) -> f64 {
        let sum_cos4: f64 = x.iter().map(|&xi| xi.cos().powi(4)).sum();
        let prod_cos2: f64 = x.iter().map(|&xi| xi.cos().powi(2)).product();
        let sum_ix2: f64 = x.iter().enumerate().map(|(i, &xi)| (i as f64 + 1.0) * xi * xi).sum();
        
        (sum_cos4 - 2.0 * prod_cos2).abs() / sum_ix2.sqrt()
    }
    fn gradient(&self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let n = x.len();
        
        let cos_vals: Vec<f64> = x.iter().map(|&xi| xi.cos()).collect();
        let sin_vals: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        
        let cos4_sum: f64 = cos_vals.iter().map(|&ci| ci.powi(4)).sum();
        let cos2_prod: f64 = cos_vals.iter().map(|&ci| ci.powi(2)).product();
        let sum_ix2: f64 = x.iter().enumerate().map(|(i, &xi)| (i as f64 + 1.0) * xi * xi).sum();
        
        let d = cos4_sum - 2.0 * cos2_prod;
        let sqrt_c = sum_ix2.sqrt();
        let sign = d.signum();
        
        let grad_a: Vec<f64> = cos_vals.iter()
            .zip(sin_vals.iter())
            .map(|(&c, &s)| -4.0 * c.powi(3) * s)
            .collect();
    
        let grad_b: Vec<f64> = x.iter().enumerate().map(|(i, _xi)| {
            if cos_vals[i].abs() < 1e-8 {
                0.0  // Avoid division by zero
            } else {
                -2.0 * cos2_prod * sin_vals[i] / cos_vals[i]
            }
        }).collect();
    
        let grad_d: Vec<f64> = grad_a.iter().zip(grad_b.iter())
            .map(|(&ga, &gb)| ga - gb)
            .collect();
    
        let grad_sqrt_c: Vec<f64> = x.iter().enumerate()
            .map(|(i, &xi)| (i as f64 + 1.0) * xi / sqrt_c)
            .collect();
    
        let grad: Vec<f64> = (0..n)
            .map(|i| sign * (grad_d[i] * sqrt_c - d * grad_sqrt_c[i]) / sum_ix2)
            .collect();
    
        Some(DVector::from_vec(grad))
    }
}

#[derive(Clone)]
struct KBFConstraints;

impl BooleanConstraintFunction<f64> for KBFConstraints {
    fn g(&self, x: &DVector<f64>) -> bool {
        let n = x.len();
        let product: f64 = x.iter().product();
        let sum: f64 = x.iter().sum();
        
        x.iter().all(|&xi| xi >= 0.0 && xi <= 10.0) &&
        product > 0.75 &&
        sum < (15.0 * n as f64) / 2.0
    }
}

// Create background contour
fn create_contour_data(obj_f: &KBF, resolution: usize) -> (Vec<Vec<f64>>, f64, f64) {
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

fn is_feasible(x: f64, y: f64) -> bool {
    let point = DVector::from_vec(vec![x, y]);
    let constraints = KBFConstraints;
    constraints.g(&point)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let config = Config {
        opt_conf: OptConf {
            max_iter: 30,
            rtol: 0.0,
            atol: 0.0,
        },
        alg_conf: AlgConf::PT(PTConf {
            num_replicas: 10,
            power_law_init: 3.0,
            power_law_final: 0.35,
            power_law_cycles: 1,
            alpha: 0.1,
            omega: 2.1,
            swap_check_type: "Always".to_string(),
            swap_frequency: 1.0,
            swap_probability: 0.8,
            mala_step_size: 0.01,
        }),
    };

    let obj_f = KBF;
    let constraints = KBFConstraints;

    // Randomly spread initial population across domain
    let mut init_pop = DMatrix::zeros(200, 2);
    for i in 0..200 {
        for j in 0..2 {
            init_pop[(i, j)] = rand::random::<f64>() * 10.0;
        }
    }

    let mut opt = NonConvexOpt::new(config, init_pop, obj_f.clone(), Some(constraints));

    let resolution = 100;
    let (z_values, min_val, max_val) = create_contour_data(&obj_f, resolution);

    let mut gif = File::create("examples/pt_kbf.gif")?;
    let mut color_palette = Vec::with_capacity(768); 
    
    color_palette.extend_from_slice(&[
        255, 0, 0,      // Bright red for population
        255, 255, 0,    // Bright yellow for best individual
    ]);
    
    // Then add grayscale colors
    for i in 0..254 {  
        color_palette.push(i as u8);    
        color_palette.push(i as u8);    
        color_palette.push(i as u8);   
    }

    let mut encoder = Encoder::new(&mut gif, 800, 800, &color_palette)?;
    encoder.set_repeat(Repeat::Infinite)?;

    for frame in 0..30 {
        let root = BitMapBackend::new("examples/pt_frame.png", (800, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(format!("PT, Keane's Bump Function - Iteration {}", frame), ("sans-serif", 30))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f64..10f64, 0f64..10f64)?;

        chart.configure_mesh()
            .disable_mesh()    
            .draw()?;

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
                
                // Draw stripes for infeasible regions
                if !is_feasible(x, y) {
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

        // Draw best individual in yellow
        let best_x = opt.get_best_individual();
        chart.draw_series(std::iter::once(
            Circle::new(
                (best_x[0], best_x[1]),
                6, 
                RGBColor(255, 255, 0).filled()
            )
        ))?;

        // Save frame
        root.present()?;
        
        // Convert PNG to GIF frame
        let img = ImageReader::open("examples/pt_frame.png")?
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
        frame.delay = 10; 
        frame.buffer = std::borrow::Cow::from(indexed_pixels);
        encoder.write_frame(&frame)?;

        opt.step();
    }

    std::fs::remove_file("examples/pt_frame.png")?;

    Ok(())
}

// Helper function to find closest color in palette
fn find_closest_color(r: u8, g: u8, b: u8, palette: &[u8]) -> usize {
    let mut best_idx = 0;
    let mut best_diff = f64::MAX;

    for i in (0..palette.len()).step_by(3) {
        let dr = r as f64 - palette[i] as f64;
        let dg = g as f64 - palette[i + 1] as f64;
        let db = b as f64 - palette[i + 2] as f64;
        let diff = dr * dr + dg * dg + db * db;

        if diff < best_diff {
            best_diff = diff;
            best_idx = i / 3;
        }
    }

    best_idx
}