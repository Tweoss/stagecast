use std::{
    fs::{self},
    path::PathBuf,
};

use egui::{emath::Numeric, Color32, DragValue, Key, ViewportBuilder};
use egui_plot::{HLine, Line, MarkerShape, Plot, Points, VLine};
use fft::{FftSample, Time, ASSUMED_SAMPLE_RATE};
use web_audio_api::{
    context::{AudioContext, BaseAudioContext},
    node::{AudioBufferSourceNode, AudioNode, AudioScheduledSourceNode, StereoPannerNode},
    AudioBuffer,
};

fn main() {
    let mut args = std::env::args();
    let input_directory = args
        .nth(1)
        .expect("pass the input directory path as an argument");
    let error_scale = args
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    let data = match (
        fs::read(PathBuf::new().join(&input_directory).join("data.cbor")),
        fs::read_to_string(PathBuf::new().join(&input_directory).join("data.json")),
    ) {
        (Ok(f), _) => serde_cbor::from_slice::<Vec<Time>>(&f).unwrap(),
        (_, Ok(f)) => serde_json::de::from_str(&f).unwrap(),
        _ => panic!("Could not read data.cbor or data.json"),
    };
    // let fft_samples = fs::read_to_string(
    //     PathBuf::new()
    //         .join(&input_directory)
    //         .join("fft_samples.json"),
    // )
    // .unwrap();
    // let fft_samples: Vec<FftSample> = serde_json::de::from_str(&fft_samples).unwrap();
    let fft_samples = vec![];
    // let data = data.iter().take(5_00).collect::<Vec<_>>();

    // Load recording.
    let context = AudioContext::default();
    let audio_file =
        std::fs::File::open(PathBuf::new().join(&input_directory).join("recording.wav"))
            .expect("No recording.wav");
    let buffer = context.decode_audio_data_sync(audio_file).unwrap();

    // Show interactive app to explore data.
    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder::default().with_maximized(true),
        ..Default::default()
    };
    eframe::run_native(
        &format!("Replay {}", &input_directory),
        native_options,
        Box::new(move |cc| {
            Box::new(ViewingApp::new(
                cc,
                data,
                fft_samples,
                context,
                buffer,
                input_directory,
                error_scale,
            ))
        }),
    )
    .unwrap();
}

struct ViewingApp {
    predicted: Vec<[f64; 2]>,
    projection: Vec<Vec<[f64; 2]>>,
    error: Vec<[f64; 2]>,
    managed_prediction: Vec<[f64; 2]>,
    dot_error: Vec<[f64; 2]>,
    fft_samples: Vec<(f64, Vec<[f64; 2]>)>,
    show_settings: ShowSettings,
    context: AudioContext,
    audio: Audio,
    selected_index: usize,
    directory_name: String,
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
            error: (Color32::RED.gamma_multiply(0.4), false),
            real_sound: true,
            predicted_sound: true,
        }
    }
}

struct Audio {
    real: (AudioBuffer, Option<AudioBufferSourceNode>, StereoPannerNode),
    predicted: (AudioBuffer, Option<AudioBufferSourceNode>, StereoPannerNode),
}

impl Audio {
    fn new(context: &AudioContext, combination: AudioBuffer) -> Self {
        let (real, predicted) = (
            combination.get_channel_data(0),
            combination.get_channel_data(1),
        );
        let make_buffer = |data: &[f32], panning: f32| {
            let mut buffer = context.create_buffer(1, data.len(), context.sample_rate());
            buffer.copy_to_channel(data, 0);
            let panner = context.create_stereo_panner();
            panner.connect(&context.destination());
            panner.pan().set_value(panning);

            (buffer, panner)
        };
        let (real, predicted) = (make_buffer(real, -1.0), make_buffer(predicted, 1.0));

        Self {
            real: (real.0, None, real.1),
            predicted: (predicted.0, None, predicted.1),
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
                let play_buffer = |buffer: AudioBuffer,
                                   panner: &StereoPannerNode,
                                   should_sound: bool|
                 -> AudioBufferSourceNode {
                    let mut src = context.create_buffer_source();
                    let duration = buffer.duration();
                    src.set_buffer(buffer);
                    if should_sound {
                        src.connect(panner);
                        // src.connect(&context.destination());
                    }
                    src.start_at_with_offset(
                        context.current_time(),
                        start_time.clamp(0.0, duration),
                    );
                    src
                };
                self.real.1 = Some(play_buffer(
                    self.real.0.clone(),
                    &self.real.2,
                    settings.real_sound,
                ));
                self.predicted.1 = Some(play_buffer(
                    self.predicted.0.clone(),
                    &self.predicted.2,
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
        fft_samples: Vec<FftSample>,
        context: AudioContext,
        combination_buffer: AudioBuffer,
        directory_name: String,
        error_scale: f64,
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
                .map(|i| select(real_iter.clone(), |t| t.projection[i].to_f64(), &data))
                .collect(),
            error: select(real_iter.clone(), |t| error_scale * t.error, &data),
            dot_error: select(real_iter.clone(), |t| error_scale * t.dot_error, &data),
            managed_prediction: select(real_iter.clone(), |t| t.managed_prediction, &data),
            fft_samples: fft_samples
                .iter()
                .map(|s| {
                    (
                        s.real,
                        s.magnitudes
                            .iter()
                            .enumerate()
                            .rev()
                            // Ignore the 10 highest frequencies
                            .skip(10)
                            // Map from period in number of samples to frequency in Hz
                            .map(|(i, v)| {
                                [(ASSUMED_SAMPLE_RATE as f64 / i as f64).log2(), *v as f64]
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
            show_settings: ShowSettings::default(),
            audio: Audio::new(&context, combination_buffer),
            context,
            selected_index: 0,
            directory_name,
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
            ui.add(
                DragValue::new(&mut self.selected_index)
                    .clamp_range(0..=(self.predicted.len() - 1)),
            );
        });

        // egui::CentralPanel::default().show(ctx, |ui| {
        let current_position = egui::TopBottomPanel::top("predictions")
            .resizable(true)
            .show(ctx, |ui| {
                // The central panel the region left after adding TopPanel's and SidePanel's
                ui.heading("Predictions");

                ui.separator();
                ui.add(egui::Slider::new(
                    &mut self.selected_index,
                    0..=(self.predicted.len() - 1),
                ));
                ui.separator();

                plot_data(ctx, ui, self)
            })
            .inner;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("FFT Samples");
            // plot_fft(ui, self, current_position);
        });
        // });
    }
}

// Return the position of the cursor in plot space or the current playback time.
fn plot_data(ctx: &egui::Context, ui: &mut egui::Ui, app: &mut ViewingApp) -> f64 {
    let response = &mut Plot::new("data_plot")
        .include_x(0.0)
        .include_y(0.0)
        .x_axis_label("real time")
        .y_axis_label("predicted time")
        .data_aspect(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(vec![
                    [0.0, 0.0],
                    [
                        app.predicted.last().unwrap()[0],
                        app.predicted.last().unwrap()[0],
                    ],
                ])
                .color(Color32::YELLOW),
            );

            if app.show_settings.predicted.1 {
                plot_ui.points(
                    Points::new(app.predicted.clone()).color(app.show_settings.predicted.0),
                );
            }
            if app.show_settings.error.1 {
                plot_ui.points(Points::new(app.error.clone()).color(app.show_settings.error.0));
                plot_ui.points(Points::new(app.dot_error.clone()).color(app.show_settings.error.0));
            }
            if app.show_settings.managed.1 {
                plot_ui.points(
                    Points::new(app.managed_prediction.clone()).color(app.show_settings.managed.0),
                );
            }
            plot_ui.points(
                Points::new(vec![app.predicted[app.selected_index]])
                    .shape(MarkerShape::Diamond)
                    .radius(15.0)
                    .color(app.show_settings.predicted.0.to_opaque()),
            );
            plot_ui.points(
                Points::new(vec![app.error[app.selected_index]])
                    .shape(MarkerShape::Circle)
                    .radius(10.0)
                    .color(app.show_settings.error.0.to_opaque()),
            );
            plot_ui.points(
                Points::new(vec![app.dot_error[app.selected_index]])
                    .shape(MarkerShape::Circle)
                    .radius(10.0)
                    .color(app.show_settings.error.0.to_opaque()),
            );
            plot_ui.points(
                Points::new(vec![app.managed_prediction[app.selected_index]])
                    .shape(MarkerShape::Circle)
                    .radius(10.0)
                    .color(app.show_settings.managed.0.to_opaque()),
            );
            if let Some(playback_position) = &app.audio.get_position() {
                ctx.request_repaint();
                plot_ui.hline(HLine::new(*playback_position));
                plot_ui.vline(VLine::new(*playback_position));
            }
        });
    let transform = response.transform;

    if let Some(pointer_pos) = response.response.hover_pos() {
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

    if let (Some(pointer_pos), true) = (
        response.response.hover_pos(),
        app.audio.get_position().is_none(),
    ) {
        transform.value_from_position(pointer_pos).x
    } else {
        app.audio.get_position().unwrap_or(0.0)
    }
}

// fn plot_fft(ui: &mut egui::Ui, app: &mut ViewingApp, current_position: f64) {
//     Plot::new("fft_plot")
//         .x_axis_label("Frequency (Hz)")
//         .y_axis_label("Amplitude")
//         .include_x(880.0)
//         .include_y(10.0)
//         .data_aspect(30.0 / 20.0)
//         .show(ui, |plot_ui| {
//             let closest_index = app
//                 .fft_samples
//                 .binary_search_by_key(&NotNan::new(current_position).unwrap(), |s| {
//                     NotNan::new(s.0).unwrap()
//                 });
//             let closest_index = match closest_index {
//                 Ok(i) => i,
//                 Err(i) => i,
//             };
//             plot_ui.line(Line::new(app.fft_samples[closest_index].1.clone()));
//         });
// }
