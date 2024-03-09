use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File};
use std::io::Write;
use std::ops::Bound::Excluded;
use std::ops::Bound::Unbounded;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use ordered_float::NotNan;
use rand::Rng;
use rand::SeedableRng;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use web_audio_api::context::{AudioContext, AudioContextRegistration, BaseAudioContext};
use web_audio_api::media_devices::MediaStreamConstraints;
use web_audio_api::media_recorder::MediaRecorder;
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ChannelConfig};
use web_audio_api::render::{AudioProcessor, AudioRenderQuantum};
use web_audio_api::{media_devices, AudioBuffer};

use fft::{
    do_fft, generate_varying_sine, load_file, random_project, FftSample, RandomVector, Time,
    DURATION, FFT_LEN, PENALTY_FRAME, QUANTUM_SIZE, REPEATS, SEARCH_DIMENSIONS,
};

/// How many previous frames of projections to look at when calculating sum
/// squared error.
const ERROR_FRAME_HISTORY_COUNT: u64 = 16;
/// Neighbor count per side per projection
const NEIGHBOR_COUNT: usize = 5;

/// Injected latency in seconds.
// const ARTIFICAL_LATENCY: f64 = 4.0;
const ARTIFICAL_LATENCY: f64 = 0.1;

/// Simple reduction of "pop" sound in the playback
/// If a predicted time is within this many frames, just don't switch.
const IGNORE_FRAME_COUNT: usize = 5;
/// Minimum time to wait between jumping between times. (1 / maximum number of "pops" per second)
const MINIMUM_SWITCH_INTERVAL: f64 = 0.2;

fn main() {
    let mut args = std::env::args();
    let output_dir = args
        .nth(1)
        .expect("pass the output directory path as an argument");
    let source_file = args
        .next()
        .expect("pass the audio source file as the second argument");
    fs::create_dir(&output_dir).expect("could not create output directory");

    let context = AudioContext::default();

    let src: Box<dyn AudioNode> = if source_file == "microphone" {
        // Take source from microphone.
        let mic = media_devices::get_user_media_sync(MediaStreamConstraints::Audio);
        // register as media element in the audio context
        Box::new(context.create_media_stream_source(&mic))
    } else if source_file == "wave" {
        let mut src = context.create_buffer_source();
        src.set_buffer(
            // generate a sample (escalating chord stack)
            generate_varying_sine(&context, DURATION),
        );
        src.set_loop(true);
        src.start();
        Box::new(src)
    } else {
        // play a buffer in a loop
        let mut src = context.create_buffer_source();
        src.set_buffer(
            // load a file
            load_file(
                &context,
                DURATION,
                // "/Users/francischua/Downloads/sample2.ogg".to_owned(),
                // "/Users/francischua/Downloads/ladispute_amp.ogg".to_owned(),
                // "/Users/francischua/Downloads/turkishmarch.ogg".to_owned(),
                // "/Users/francischua/Downloads/stravinsky_cut.ogg".to_owned(),
                source_file,
            ),
        );
        src.set_loop(true);
        src.start();
        Box::new(src)
    };

    let fft = FftNode::new(&context, FftOptions { fft_len: FFT_LEN });
    let (fft, prediction_rx) = fft.consume_receiver();

    let left_panner = context.create_stereo_panner();
    left_panner.connect(&context.destination());
    left_panner.pan().set_value(1.);

    let delayed_input = context.create_delay(ARTIFICAL_LATENCY);
    src.connect(&delayed_input);
    delayed_input.connect(&fft);

    let context = Arc::new(context);
    let playback = PlaybackNode::new(context.clone(), PlaybackOptions { prediction_rx });
    delayed_input.connect(&playback);

    let right_panner = context.create_stereo_panner();
    right_panner.connect(&context.destination());
    right_panner.pan().set_value(-1.);
    src.connect(&right_panner);
    playback.connect(&left_panner);
    // use this line instead of the above two when playing live
    // playback.connect(&context.destination());

    let output = context.create_media_stream_destination();
    let recorder = MediaRecorder::new(output.stream());
    right_panner.connect(&output);
    left_panner.connect(&output);
    let recording = Box::<Arc<_>>::leak(Box::new(Arc::new(Mutex::new(Vec::new()))));
    recorder.set_ondataavailable(|mut event| {
        recording.lock().unwrap().append(&mut event.blob);
    });
    {
        let recording = recording.clone();
        let output_dir = output_dir.clone();
        recorder.set_onstop(move |_| {
            File::create(PathBuf::new().join(&output_dir).join("recording.wav"))
                .unwrap()
                .write_all(&recording.lock().unwrap())
                .unwrap();
        });
    }
    recorder.start();

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs_f32(
        DURATION * REPEATS as f32,
    ));

    println!("Done running, writing data out.");

    recorder.stop();

    let times = fft.times.lock().unwrap().to_vec();
    serde_json::to_writer(
        &File::create(PathBuf::new().join(&output_dir).join("data.json")).unwrap(),
        &times,
    )
    .unwrap();

    let fft_samples = fft.samples.lock().unwrap().to_vec();
    serde_json::to_writer(
        &File::create(PathBuf::new().join(&output_dir).join("fft_samples.json")).unwrap(),
        &fft_samples,
    )
    .unwrap();

    make_graph(&times, &PathBuf::new().join(&output_dir).join("graph.png"));
}

struct FftNode<PredictionType> {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    times: Arc<Mutex<Vec<Time>>>,
    samples: Arc<Mutex<Vec<FftSample>>>,
    prediction: PredictionType,
}

struct FftOptions {
    fft_len: usize,
}

impl FftNode<Receiver<Prediction>> {
    fn new(context: &impl BaseAudioContext, options: FftOptions) -> Self {
        context.register(move |registration| {
            let real_planner: RealFftPlanner<f32> = RealFftPlanner::<f32>::new();
            let times = Arc::new(Mutex::new(vec![]));
            let samples = Arc::new(Mutex::new(vec![]));
            // Deterministic for reproducibility.
            let mut rng = rand::rngs::SmallRng::seed_from_u64(0xDEADBEEF);
            let mut gen_random_vector = || {
                let mut vec = (0..(options.fft_len / 2 + 1))
                    .map(|_| rng.gen_range(-1.0_f32..1.0))
                    .collect::<Vec<_>>();
                // Normalize to unit.
                let magnitude = vec.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
                for el in &mut vec {
                    *el /= magnitude;
                }

                RandomVector { points: vec }
            };
            // Set exact size of backing buffer.
            let mut past_input: VecDeque<_> = vec![].into();
            past_input.reserve_exact(options.fft_len);
            past_input.append(&mut vec![0.0f32; options.fft_len].into());

            let (tx, rx) = mpsc::channel();

            let render = FftRenderer {
                sample_rate: context.sample_rate() as f64,
                planner: real_planner,
                past_input,
                data: vec![Complex::new(0.0, 0.0); options.fft_len / 2 + 1],
                random_vector: (0..SEARCH_DIMENSIONS)
                    .map(|_| gen_random_vector())
                    .collect(),
                frame_data: FrameData {
                    btree: (0..SEARCH_DIMENSIONS).map(|_| BTreeMap::new()).collect(),
                    projections: HashMap::new(),
                    frames: vec![],
                },
                times: times.clone(),
                samples: samples.clone(),
                last_prediction: (0, tx),
                prediction_manager: PredictionManager { data: vec![] },
            };
            let node = FftNode {
                registration,
                channel_config: ChannelConfig::default(),
                times,
                samples,
                prediction: rx,
            };

            (node, Box::new(render))
        })
    }

    fn consume_receiver(self) -> (FftNode<()>, Receiver<Prediction>) {
        let rx = self.prediction;
        (
            FftNode {
                registration: self.registration,
                channel_config: self.channel_config,
                times: self.times,
                samples: self.samples,
                prediction: (),
            },
            rx,
        )
    }
}

impl<T> AudioNode for FftNode<T> {
    fn registration(&self) -> &web_audio_api::context::AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &web_audio_api::node::ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        0
    }
}

struct FftRenderer {
    sample_rate: f64,
    past_input: VecDeque<f32>,
    data: Vec<Complex<f32>>,
    planner: RealFftPlanner<f32>,
    random_vector: Vec<RandomVector>,
    frame_data: FrameData,
    times: Arc<Mutex<Vec<Time>>>,
    samples: Arc<Mutex<Vec<FftSample>>>,
    last_prediction: (u64, Sender<Prediction>),
    prediction_manager: PredictionManager,
}

struct FrameData {
    /// Map from random projection to frame number
    btree: Vec<BTreeMap<NotNan<f32>, u64>>,
    /// Map from timestamp to index in frame_history and projections.
    projections: HashMap<u64, (usize, Vec<f64>)>,
    /// List of consecutive frame numbers
    frames: Vec<u64>,
}

impl AudioProcessor for FftRenderer {
    fn process(
        &mut self,
        inputs: &[web_audio_api::render::AudioRenderQuantum],
        _outputs: &mut [web_audio_api::render::AudioRenderQuantum],
        _params: web_audio_api::render::AudioParamValues<'_>,
        scope: &web_audio_api::render::RenderScope,
    ) -> bool {
        // Use a vecdeque to eliminate rotating costs.
        let sample_len = inputs[0].channel_data(0).len();
        self.past_input.rotate_left(sample_len);
        let start_index = self.past_input.len() - sample_len;
        // Copy input in. (TODO: maybe reuse past FFT's)
        self.past_input
            .range_mut(start_index..)
            .zip(inputs[0].channel_data(0).iter())
            .for_each(|(i, s)| *i = *s);

        // TODO try scratch space, storing Fft after planning
        do_fft(
            &mut self.planner,
            self.past_input.clone().make_contiguous(),
            &mut self.data,
        );
        // Send some fft samples every so often.
        {
            let mut samples = self.samples.lock().unwrap();
            if !samples
                .last()
                // Every 0.05 seconds.
                .is_some_and(|s| scope.current_time - s.real < 0.05)
            {
                samples.push(FftSample {
                    real: scope.current_time,
                    magnitudes: self
                        .data
                        .iter()
                        .map(|c| c.norm())
                        .collect::<Vec<_>>()
                        .clone(),
                })
            }
        }

        let current_projections: Vec<_> = self
            .random_vector
            .iter()
            .map(|v| random_project(&self.data, v) as f64)
            .collect();
        let current_frame_index = {
            self.frame_data.frames.push(scope.current_frame);
            self.frame_data.frames.len() - 1
        };

        self.frame_data.projections.insert(
            scope.current_frame,
            (current_frame_index, current_projections.clone()),
        );

        // Find the best guess among neighbors.
        // TODO: try R-tree
        if let Some((error, best_frame)) = current_projections
            .iter()
            .zip(self.frame_data.btree.iter_mut())
            .flat_map(|(p, b)| {
                let projection = NotNan::new(*p as f32).unwrap();
                b.insert(projection, scope.current_frame);
                let neighbors = b
                    .range(..projection)
                    // TODO remove (hacky). find better way of penalizing.
                    .filter(|(_, n_f)| (scope.current_frame - *n_f) > PENALTY_FRAME)
                    .rev()
                    .take(NEIGHBOR_COUNT)
                    .chain(
                        // Make sure not to include the *just* inserted projection.
                        b.range((Excluded(projection), Unbounded))
                            .filter(|(_, n_f)| (scope.current_frame - *n_f) > PENALTY_FRAME)
                            .take(NEIGHBOR_COUNT),
                    )
                    .map(|(_, t)| *t);
                neighbors
            })
            .map(|n_frame| -> (NotNan<f64>, u64) {
                // Using sum squared error over all projections.
                (
                    NotNan::new(calculate_error(
                        &self.frame_data.projections,
                        &self.frame_data.frames,
                        n_frame,
                        scope.current_frame,
                        self.last_prediction.0,
                    ))
                    .unwrap(),
                    n_frame,
                )
            })
            .min_by_key(|(error, _)| *error)
        {
            let predicted_time = best_frame as f64 / self.sample_rate;
            self.prediction_manager
                .add_prediction(predicted_time, scope.current_time);
            let (frame, prediction) = self
                .prediction_manager
                .get_prediction(scope.current_time, self.sample_rate);
            let _ = self.last_prediction.1.send(prediction.clone());
            self.last_prediction.0 = frame;

            self.times.lock().unwrap().push(Time {
                real: scope.current_time,
                predicted: predicted_time,
                projection: current_projections.clone(),
                error: *error,
                managed_prediction: prediction.predicted_time,
            });
        }

        false
    }
}

// TODO: perhaps weight more recent times more heavily
// TODO: use r-trees?
fn calculate_error(
    projections: &HashMap<u64, (usize, Vec<f64>)>,
    frames: &[u64],
    neighbor_frame: u64,
    current_frame: u64,
    last_prediction: u64,
) -> f64 {
    // Using sum squared error over all projections.
    let current_index = projections
        .get(&current_frame)
        .expect("should have stored current frame's projection")
        .0;
    let neighbor_index = projections
        .get(&neighbor_frame)
        .expect("should have neighbor's projection")
        .0;

    // Take the frames such that their FFT's don't overlap
    let it = frames[..=neighbor_index]
        .iter()
        .rev()
        .step_by(FFT_LEN / QUANTUM_SIZE as usize)
        .zip(
            frames[..=current_index]
                .iter()
                .rev()
                .step_by(FFT_LEN / QUANTUM_SIZE as usize),
        )
        .take(ERROR_FRAME_HISTORY_COUNT as usize)
        .flat_map(|(n_i, c_i)| {
            projections
                .get(n_i)
                .unwrap()
                .1
                .iter()
                .zip(projections.get(c_i).unwrap().1.iter())
        })
        .map(|(n, c)| (n - c).powi(2));

    assert_ne!(
        it.clone().count(),
        0,
        "Number of comparable frames should never be 0."
    );

    // let frame_delta = current_frame.abs_diff(neighbor_frame);

    // // Add a bias towards more recent samples.
    // // 4410000 => about 1 additional error every 100 seconds.
    // // TODO make into a constant
    // error + (current_frame - neighbor_frame) as f64 / 4_410_000.

    // if frame_delta < PENALTY_FRAME {
    //     // Scale by log so positive infinity near current frames, and PENALTY near PENALTY_FRAME
    //     // Add 1 to avoid logarithm of 0.
    //     let ratio = (frame_delta + 1) as f64 / (PENALTY_FRAME + 1) as f64;
    //     // Add 1 to make value PENALTY near PENALTY_FRAME
    //     let penalty = PENALTY * (ratio.log10().abs() + 1.);
    //     assert!(penalty >= PENALTY, "{}", penalty);
    //     // Add some error for nearby times
    //     error += penalty;
    // }

    // // TODO handle last predicted time
    // if last_prediction.abs_diff(neighbor_frame) < BONUS_FRAME {
    //     // Decrease error for times near last predicted time
    //     error /= BONUS
    // }

    // Add a temporary hack error
    // TODO: remove / replace with something nicer
    (it.clone().sum::<f64>() / (it.count() as f64)).sqrt()
        + if (current_frame).abs_diff(neighbor_frame) < 100_000 {
            0.04
        } else {
            0.0
        }
}

struct PlaybackNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
}

struct PlaybackOptions {
    prediction_rx: Receiver<Prediction>,
}

impl PlaybackNode {
    fn new(context: Arc<AudioContext>, options: PlaybackOptions) -> Self {
        context.clone().register(move |registration| {
            let render = PlaybackRenderer {
                context: context.clone(),
                past_data: ResizingBuffer::new(context),
                prediction: options.prediction_rx,
            };
            let node = PlaybackNode {
                registration,
                channel_config: ChannelConfig::default(),
            };

            (node, Box::new(render))
        })
    }
}

impl AudioNode for PlaybackNode {
    fn registration(&self) -> &web_audio_api::context::AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &web_audio_api::node::ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }
}

struct PlaybackRenderer {
    context: Arc<AudioContext>,
    past_data: ResizingBuffer,
    prediction: Receiver<Prediction>,
}

#[derive(Clone)]
struct Prediction {
    predicted_time: f64,
    time_of_prediction: f64,
}

struct ResizingBuffer {
    buffer: AudioBuffer,
    used: usize,
    start_time: f64,
    playback_index: usize,
    /// Hard ceiling on how often we can switch.
    last_switch_time: f64,
}

/// Avoid skipping back and forth rapidly.
struct PredictionManager {
    data: Vec<(f64, NotNan<f64>)>,
}

impl PredictionManager {
    /// The range in which two timestamps are considered close enough to be the same prediction.
    const EQUAL_WINDOW: f64 = 1.0;
    const DECAY_RATIO: f64 = 21.0 / 22.0;
    const INCREMENT: f64 = 1.0;

    /// Returns a prediction and the closest frame that matches that prediction.
    /// Assumes that frames are always divisible by QUANTUM_SIZE.
    fn add_prediction(&mut self, predicted_time: f64, current_time: f64) {
        let normalized_time = predicted_time - current_time;
        let mut found_match = false;
        for (time, count) in &mut self.data {
            // 1 second window
            if (*time - normalized_time).abs() < Self::EQUAL_WINDOW {
                // TODO: just rewrite?
                // *time = **count;
                // Recalculate the prediction, using the count as a weight.
                *time = (*time * **count + normalized_time) / (**count + 1.0);
                *count += Self::INCREMENT;
                found_match = true;
            } else {
                *count *= Self::DECAY_RATIO;
            }
        }

        if !found_match {
            self.data.push((normalized_time, NotNan::new(1.0).unwrap()));
        }

        let best_time = self.get_best_time(current_time);
        // Remove everything (except the best time) with a counter value less than 1.
        self.data
            .retain(|(t, c)| **c >= 1.0 && t.partial_cmp(&best_time) != Some(Ordering::Equal));
    }

    fn get_prediction(&self, current_time: f64, sample_rate: f64) -> (u64, Prediction) {
        let best_time = self.get_best_time(current_time);

        (
            Self::round_to_frame(best_time, sample_rate),
            Prediction {
                predicted_time: best_time,
                time_of_prediction: current_time,
            },
        )
    }

    /// Assumes that frames are always divisible by QUANTUM_SIZE.
    fn round_to_frame(time: f64, sample_rate: f64) -> u64 {
        (time * sample_rate / QUANTUM_SIZE as f64).round() as u64 * QUANTUM_SIZE
    }

    fn get_best_time(&self, current_time: f64) -> f64 {
        self.data
            .iter()
            .max_by_key(|d| d.1)
            .map(|d| d.0)
            .unwrap_or(0.0)
            + current_time
    }
}

impl ResizingBuffer {
    // 64 quantums
    const DEFAULT_SIZE: usize = 128 * 64;

    fn new(context: Arc<AudioContext>) -> Self {
        Self {
            buffer: context.create_buffer(1, Self::DEFAULT_SIZE, context.sample_rate()),
            used: 0,
            start_time: context.current_time(),
            playback_index: 0,
            last_switch_time: 0.0,
        }
    }

    fn append(&mut self, context: Arc<AudioContext>, input: &AudioRenderQuantum) {
        let input = input.channel_data(0);
        // If out of capacity, double size of buffer.
        if self.used + input.len() >= self.buffer.length() {
            let mut new_buffer =
                context.create_buffer(1, self.buffer.length() * 2, context.sample_rate());
            new_buffer.copy_to_channel(self.buffer.get_channel_data(0), 0);
            self.buffer = new_buffer;
        }
        // Copy in new data.
        self.buffer.copy_to_channel_with_offset(input, 0, self.used);
        self.used += input.len();
    }

    /// Sets the new time if it is far away.
    fn try_set_time(&mut self, time: f64, sample_len: usize, current_time: f64) -> bool {
        let sample_duration = sample_len as f64 / self.buffer.sample_rate() as f64;
        // This prevents small amounts of noise causing lots of static.
        // However, it also means that if the prediction is being incrementally
        // corrected (perhaps due to changes in tempo), then there would be
        // additional latency until `try_set_time` reflects that correction.
        let time_tolerance = IGNORE_FRAME_COUNT as f64 * sample_duration;
        let new_time = if (self.get_time() - time).abs() < time_tolerance
            || (current_time - self.last_switch_time).abs() < MINIMUM_SWITCH_INTERVAL
        {
            return false;
        } else {
            self.last_switch_time = current_time;
            time
        };

        let time_offset = new_time - self.start_time;
        let new_index = (time_offset * self.buffer.sample_rate() as f64) as usize;
        self.playback_index = new_index;
        true
    }

    fn get_time(&self) -> f64 {
        self.start_time + (self.playback_index as f64 / self.buffer.sample_rate() as f64)
    }

    fn play(&mut self, output: &mut AudioRenderQuantum) {
        let output = output.channel_data_mut(0);
        let written_count = output
            .iter_mut()
            .zip(
                // self.buffer.get_channel_data(0)[self.playback_index..].iter(),
                // Avoid panic when not enough data exists by using iterator methods.
                self.buffer
                    .get_channel_data(0)
                    .iter()
                    .skip(self.playback_index.saturating_sub(1)),
            )
            .map(|(o, b)| *o = *b)
            .count();
        self.playback_index += written_count;
        // If the written_count is 0, then the predicted time is in front of the data we have. So, play the most recent data we have
        //
        // note: this sounds pretty bad when the latency is in the seconds because the good predictions (perhaps repeats within a piece)
        // will interleave with audio from seconds ago. should be okay with latency in the milliseconds though.
        if written_count == 0 {
            let count = output.len();
            output
                .iter_mut()
                .zip(
                    self.buffer
                        .get_channel_data(0)
                        .iter()
                        .take(self.used)
                        .rev()
                        .take(count)
                        .rev(),
                )
                .for_each(|(o, b)| *o = *b);
        }
    }
}

impl AudioProcessor for PlaybackRenderer {
    fn process(
        &mut self,
        inputs: &[web_audio_api::render::AudioRenderQuantum],
        outputs: &mut [web_audio_api::render::AudioRenderQuantum],
        _params: web_audio_api::render::AudioParamValues<'_>,
        scope: &web_audio_api::render::RenderScope,
    ) -> bool {
        self.past_data.append(self.context.clone(), &inputs[0]);
        if let Ok(new_prediction) = self.prediction.try_recv() {
            // Account for possible elapsed time between sending and receiving.
            let updated_time = (scope.current_time - new_prediction.time_of_prediction)
                + new_prediction.predicted_time;
            // TODO check logic of latency
            // + ARTIFICAL_LATENCY;

            self.past_data.try_set_time(
                updated_time,
                inputs[0].channel_data(0).len(),
                scope.current_time,
            );
        }
        // // Only play if much in the past.
        // if scope.current_time - self.past_data.get_time() > DURATION as f64 {
        //     self.past_data.play(&mut outputs[0]);
        // }
        self.past_data.play(&mut outputs[0]);

        false
    }
}

fn make_graph<P: AsRef<Path>>(data: &[Time], output_file: &P) {
    use plotters::prelude::*;
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
    let root = BitMapBackend::new(&output_file, (1920, 1080)).into_drawing_area();
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
}
