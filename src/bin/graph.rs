use std::fs::{self};

use fft::{Time, DURATION, REPEATS, SEARCH_DIMENSIONS};
use ordered_float::NotNan;

fn main() {
    use plotters::prelude::*;
    let data = fs::read_to_string("output/data.json").unwrap();
    let data: Vec<Time> = serde_json::de::from_str(&data).unwrap();

    let root = BitMapBackend::new("output/graph.png", (1920, 1080)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0.0..(DURATION as f64 * REPEATS as f64),
            -8.2f64..(DURATION as f64 * REPEATS as f64),
        )
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(
            data.iter()
                .map(|t| Circle::new((t.real, t.predicted), 2, GREEN.filled())),
        )
        .unwrap()
        .label("predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    // Fit the projection data max into DURATION * REPEATS / 2 height
    let all_data = data
        .iter()
        .flat_map(|t| t.projection.iter().map(|p| NotNan::new(*p).unwrap()));
    let max = all_data.clone().max().expect("some maximum");
    let scale = DURATION as f64 * REPEATS as f64 / *max / 2.;

    for i in 0..SEARCH_DIMENSIONS {
        chart
            .draw_series(LineSeries::new(
                data.iter().map(|t| (t.real, scale * t.projection[i])),
                &BLUE.mix(0.1),
            ))
            .unwrap()
            .label(&format!("projections (scaled by {})", scale))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .filter(|t| t.real < DURATION as f64 * (REPEATS - 1) as f64)
                    .map(|t| (t.real + DURATION as f64, scale * t.projection[i])),
                &RED.mix(0.1),
            ))
            .unwrap()
            .label("projections - 2sec")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}
