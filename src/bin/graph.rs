use std::fs::{self};

use egui::{Color32, Key};
use egui_plot::{HLine, Plot, Points, VLine};
use ordered_float::NotNan;

use fft::{Time, DURATION, REPEATS};
use web_audio_api::{
    context::{AudioContext, BaseAudioContext},
    node::{AudioBufferSourceNode, AudioNode, AudioScheduledSourceNode},
    AudioBuffer,
};

fn main() {
    use plotters::prelude::*;
    let input_file = std::env::args()
        .nth(1)
        .expect("pass the input json path as an argument");
    let data = fs::read_to_string(input_file).unwrap();
    let data: Vec<Time> = serde_json::de::from_str(&data).unwrap();
    // let data = data.iter().take(5_00).collect::<Vec<_>>();

    // Debug info
    let it = data
        .iter()
        .map(|t| NotNan::new(t.predicted - t.real).unwrap());
    dbg!(it.clone().max());
    dbg!(it.clone().min());
    dbg!(it.clone().map(|f| *f).sum::<f64>() / it.count() as f64);
    let it = data.iter().map(|t| NotNan::new(t.error).unwrap());
    dbg!(it.clone().max());
    dbg!(it.clone().min());
    dbg!(it.clone().map(|f| *f).sum::<f64>() / it.count() as f64);

    // Draw onto an image and output.
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
                .map(|t| Circle::new((t.real, t.predicted), 1, GREEN.mix(0.03))),
        )
        .unwrap()
        .label("predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    chart
        .draw_series(
            data.iter()
                .map(|t| Circle::new((t.real, t.managed_prediction), 1, MAGENTA.mix(0.1))),
        )
        .unwrap()
        .label("manager predicted")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], MAGENTA));
    chart
        .draw_series(
            data.iter()
                .map(|t| Circle::new((t.real, t.error), 1, RED.mix(0.1))),
        )
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();

    // Load recording.
    let context = AudioContext::default();
    let audio_file = std::fs::File::open("output/recording.wav").expect("No output/recording.wav");
    let buffer = context.decode_audio_data_sync(audio_file).unwrap();

    // Show interactive app to explore data.
    let native_options = eframe::NativeOptions {
        ..Default::default()
    };
    eframe::run_native(
        "data viewer",
        native_options,
        Box::new(|cc| Box::new(ViewingApp::new(cc, data, context, buffer))),
    )
    .unwrap();
}

struct ViewingApp {
    predicted: Vec<[f64; 2]>,
    projection: Vec<Vec<[f64; 2]>>,
    error: Vec<[f64; 2]>,
    managed_prediction: Vec<[f64; 2]>,
    show_settings: ShowSettings,
    context: AudioContext,
    audio: Audio,
}

struct ShowSettings {
    managed: (Color32, bool),
    error: (Color32, bool),
    predicted: (Color32, bool),
    real_sound: bool,
    predicted_sound: bool,
}

impl Default for ShowSettings {
    fn default() -> Self {
        ShowSettings {
            managed: (Color32::from_rgb(198, 88, 201).gamma_multiply(0.2), true),
            predicted: (Color32::GREEN.gamma_multiply(0.2), true),
            error: (Color32::RED, false),
            real_sound: true,
            predicted_sound: true,
        }
    }
}

struct Audio {
    real: (AudioBuffer, Option<AudioBufferSourceNode>),
    predicted: (AudioBuffer, Option<AudioBufferSourceNode>),
}

impl Audio {
    fn new(context: &AudioContext, combination: AudioBuffer) -> Self {
        let (real, predicted) = (
            combination.get_channel_data(0),
            combination.get_channel_data(1),
        );
        let make_buffer = |data: &[f32]| {
            let mut buffer = context.create_buffer(1, data.len(), context.sample_rate());
            buffer.copy_to_channel(data, 0);
            buffer
        };
        let (real, predicted) = (make_buffer(real), make_buffer(predicted));

        Self {
            real: (real, None),
            predicted: (predicted, None),
        }
    }

    fn get_position(&self) -> Option<f64> {
        // The times should be the same.
        match (&self.real.1, &self.predicted.1) {
            (Some(r), _) => Some(r.position()),
            (None, Some(p)) => Some(p.position()),
            (None, None) => None,
        }
    }

    fn toggle_play(&mut self, context: &AudioContext, start_time: f64, settings: &ShowSettings) {
        match (&mut self.real.1, &mut self.predicted.1) {
            (Some(r), Some(p)) => {
                r.stop();
                p.stop();
                self.real.1 = None;
                self.predicted.1 = None;
            }
            (Some(r), None) => {
                r.stop();
                self.real.1 = None;
            }
            (None, Some(p)) => {
                p.stop();
                self.predicted.1 = None;
            }
            (None, None) => {
                let play_buffer =
                    |buffer: AudioBuffer, should_sound: bool| -> AudioBufferSourceNode {
                        let mut src = context.create_buffer_source();
                        src.set_buffer(buffer);
                        if should_sound {
                            src.connect(&context.destination());
                        }
                        src.start_at_with_offset(context.current_time(), start_time);
                        src
                    };
                self.real.1 = Some(play_buffer(self.real.0.clone(), settings.real_sound));
                self.predicted.1 = Some(play_buffer(
                    self.predicted.0.clone(),
                    settings.predicted_sound,
                ));
            }
        }
    }
}

impl ViewingApp {
    /// Called once before the first frame.
    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        data: Vec<Time>,
        context: AudioContext,
        combination_buffer: AudioBuffer,
    ) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        let real_iter = data.iter().map(|t| t.real);

        fn select(
            real: impl Iterator<Item = f64>,
            selector: impl Fn(&Time) -> f64,
            data: &[Time],
        ) -> Vec<[f64; 2]> {
            real.zip(data.iter().map(selector))
                .map(|(r, d)| [r, d])
                .collect()
        }

        ViewingApp {
            predicted: select(real_iter.clone(), |t| t.predicted, &data),
            projection: (0..data[0].projection.len())
                .map(|i| select(real_iter.clone(), |t| t.projection[i], &data))
                .collect(),
            error: select(real_iter.clone(), |t| t.error, &data),
            managed_prediction: select(real_iter.clone(), |t| t.managed_prediction, &data),
            show_settings: ShowSettings::default(),
            audio: Audio::new(&context, combination_buffer),
            context,
        }
    }
}

impl eframe::App for ViewingApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                egui::widgets::global_dark_light_mode_buttons(ui);
            });
        });

        egui::SidePanel::left("left_panel").show(ctx, |ui| {
            ui.checkbox(&mut self.show_settings.predicted.1, "Predicted");
            ui.color_edit_button_srgba(&mut self.show_settings.predicted.0);
            ui.checkbox(&mut self.show_settings.managed.1, "Managed");
            ui.color_edit_button_srgba(&mut self.show_settings.managed.0);
            ui.checkbox(&mut self.show_settings.error.1, "Error");
            ui.color_edit_button_srgba(&mut self.show_settings.error.0);
            ui.checkbox(&mut self.show_settings.real_sound, "Play Real");
            ui.checkbox(&mut self.show_settings.predicted_sound, "Play Predicted");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("eframe template");

            ui.separator();

            plot(ctx, ui, self);

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                egui::warn_if_debug_build(ui);
            });
        });
    }
}

fn plot(ctx: &egui::Context, ui: &mut egui::Ui, app: &mut ViewingApp) {
    let transform = Plot::new("custom_axes")
        .clamp_grid(true)
        .show(ui, |plot_ui| {
            if app.show_settings.predicted.1 {
                plot_ui.points(
                    Points::new(app.predicted.clone()).color(app.show_settings.predicted.0),
                );
            }
            if app.show_settings.error.1 {
                plot_ui.points(Points::new(app.error.clone()).color(app.show_settings.error.0));
            }
            if app.show_settings.managed.1 {
                plot_ui.points(
                    Points::new(app.managed_prediction.clone()).color(app.show_settings.managed.0),
                );
            }
            if let Some(playback_position) = &app.audio.get_position() {
                ctx.request_repaint();
                plot_ui.hline(HLine::new(*playback_position));
                plot_ui.vline(VLine::new(*playback_position));
            }
        })
        .transform;

    if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
        let start_time = if ui.input_mut(|i| i.key_pressed(Key::Space) && i.modifiers.shift_only())
        {
            Some(transform.value_from_position(pointer_pos).y)
        } else if ui.input_mut(|i| i.key_pressed(Key::Space) && i.modifiers.is_none()) {
            Some(transform.value_from_position(pointer_pos).x)
        } else {
            None
        };
        if let Some(start_time) = start_time {
            app.audio
                .toggle_play(&app.context, start_time, &app.show_settings);
        }
    }
}
