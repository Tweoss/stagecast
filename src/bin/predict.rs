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
    do_fft, generate_varying_sine, load_file, FftSample, RandomSubspace, Time, DURATION, FFT_LEN,
    IGNORE_FRAME, PROJECTION_LENGTH, QUANTUM_SIZE, REPEATS, SEARCH_DIMENSIONS,
};

/// How many previous frames of projections to look at when calculating sum
/// squared error.
const ERROR_FRAME_HISTORY_COUNT: u64 = 16;
/// Neighbor count per side per projection
const NEIGHBOR_COUNT: usize = 5;
/// The maximum distance the dot product can be for a prediction to be
/// considered good.
const DOT_LIMIT: f64 = 4.0;

/// Injected latency in seconds.
const ARTIFICIAL_LATENCY: f64 = 0.1;

/// Simple reduction of "pop" sound in the playback
/// If a predicted time is within this many frames, just don't switch.
const IGNORE_FRAME_COUNT: usize = 5;
/// Minimum time to wait between jumping between times. (1 / maximum number of "pops" per second)
const MINIMUM_SWITCH_INTERVAL: f64 = 0.2;

/// How much importance falls off from recent samples to less recent samples.
const WEIGHT_FALLOFF: f64 = 0.8;

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

    // TODO: take duration as arg
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
            load_file(&context, DURATION, source_file),
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

    let delayed_input = context.create_delay(ARTIFICIAL_LATENCY + 1.0);
    delayed_input
        .delay_time()
        .set_value(ARTIFICIAL_LATENCY as f32);
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
    delayed_input.disconnect_from(&playback);
    // TODO disconnect src from output.
    delayed_input.disconnect_from(&fft);
    playback.disconnect_from(&left_panner);

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
            let mut gen_random_subspace = || {
                let mut vec = (0..(options.fft_len / 2 + 1))
                    .map(|_| rng.gen_range(-1.0_f32..1.0))
                    .collect::<Vec<_>>();
                // Normalize to unit.
                let magnitude = vec.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
                for el in &mut vec {
                    *el /= magnitude;
                }

                RandomSubspace { points: vec }
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
                random_subspace: gen_random_subspace(),
                frame_data: FrameData {
                    btree: (0..FFT_LEN / 400).map(|_| BTreeMap::new()).collect(),
                    projections: HashMap::new(),
                    frames: vec![],
                },
                times: times.clone(),
                samples: samples.clone(),
                last_prediction: (None, tx),
                prediction_manager: PredictionManager {
                    data: vec![],
                    last_prediction: None,
                },
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
    random_subspace: RandomSubspace<{ PROJECTION_LENGTH }>,
    frame_data: FrameData,
    times: Arc<Mutex<Vec<Time>>>,
    samples: Arc<Mutex<Vec<FftSample>>>,
    last_prediction: (Option<u64>, Sender<Prediction>),
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
            // let mut samples = self.samples.lock().unwrap();
            // // Every .1 seconds
            // if (scope.current_time * 100.0).rem_euclid(100.0) < 1.0 {
            //     samples.push(FftSample {
            //         real: scope.current_time,
            //         magnitudes: self
            //             .data
            //             .iter()
            //             .map(|c| c.norm())
            //             .collect::<Vec<_>>()
            //             .clone(),
            //     })
            // }
        }

        let current_projections: Vec<_> = self.random_subspace.random_project(&self.data);
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
                b.range(..projection)
                    .filter(|(_, n_f)| (scope.current_frame - *n_f) > IGNORE_FRAME)
                    .rev()
                    .take(NEIGHBOR_COUNT)
                    .chain(
                        // Make sure not to include the *just* inserted projection.
                        b.range((Excluded(projection), Unbounded))
                            .filter(|(_, n_f)| (scope.current_frame - *n_f) > IGNORE_FRAME)
                            .take(NEIGHBOR_COUNT),
                    )
                    .map(|(_, t)| *t)
            })
            // Include the last prediction (if it exists and is far enough away)
            .chain(
                self.last_prediction
                    .0
                    .map(|v| v + QUANTUM_SIZE)
                    .filter(|p| {
                        self.frame_data.projections.get(p).is_some()
                            && scope.current_frame - p > IGNORE_FRAME
                    }),
            )
            .map(|n_frame| -> (NotNan<f64>, u64) {
                // General logic: if the best (by SSE) option that is not recent has abs(ln(dot)) <= abs(ln(DOT_LIMIT)), then
                // we say it is a good prediction and use that. otherwise, we default to saying the audio is "new" content and
                // predict the most recent time. This *might* result in lots of recent times => manager might just say
                // everything is recent => we might need to handle.

                // Using sum squared error over all projections.
                (
                    NotNan::new(calculate_sse(
                        generate_iterator(
                            &self.frame_data.projections,
                            &self.frame_data.frames,
                            n_frame,
                        ),
                        generate_iterator(
                            &self.frame_data.projections,
                            &self.frame_data.frames,
                            scope.current_frame,
                        ),
                    ))
                    .unwrap(),
                    n_frame,
                )
            })
            .min_by_key(|(error, _)| *error)
        {
            let dot_error = calculate_scaled_dot(
                generate_iterator(
                    &self.frame_data.projections,
                    &self.frame_data.frames,
                    best_frame,
                ),
                generate_iterator(
                    &self.frame_data.projections,
                    &self.frame_data.frames,
                    scope.current_frame,
                ),
            );
            let predicted_time = if error.is_finite() && dot_error < DOT_LIMIT.ln() {
                best_frame as f64 / self.sample_rate + ARTIFICIAL_LATENCY
            } else {
                scope.current_time
            };
            self.prediction_manager
                .add_prediction(predicted_time, scope.current_time);
            let (frame, prediction) = self
                .prediction_manager
                .get_prediction(scope.current_time, self.sample_rate);
            let _ = self.last_prediction.1.send(prediction.clone());
            self.last_prediction.0 = Some(frame);

            self.times.lock().unwrap().push(Time {
                real: scope.current_time,
                predicted: predicted_time - ARTIFICIAL_LATENCY,
                projection: current_projections.clone(),
                error: *error,
                managed_prediction: prediction.predicted_time - ARTIFICIAL_LATENCY,
                dot_error,
            });
        }

        false
    }
}

// TODO: perhaps split into frequencies and only project relevant ones, ie 440*2^(-3, +3)
// TODO: maybe have one long fft and one short fft. that way we match the rough place
// and the precise place both. then both are somewhat irrespective of tempo, but having
// the long one means we have more context, and the short one means we choose the best
// place locally.
// TODO: use r-trees?
fn generate_iterator<'a>(
    projections: &'a HashMap<u64, (usize, Vec<f64>)>,
    frames: &'a [u64],
    frame: u64,
) -> impl Iterator<Item = &'a f64> + Clone {
    let index = projections
        .get(&frame)
        .expect("should have stored frame's projection")
        .0;

    frames[..=index]
        .iter()
        .rev()
        // Take the frames such that their FFT input windows don't overlap
        .step_by(FFT_LEN / QUANTUM_SIZE as usize)
        .flat_map(|c_i| projections.get(c_i).unwrap().1.iter())
        // If there are not enough frames, don't just give back 0 error.
        // Instead, fill with 0's so that error can still be calculated against
        // the current_index's past.
        .chain(std::iter::repeat(&0.0))
        .take(ERROR_FRAME_HISTORY_COUNT as usize * SEARCH_DIMENSIONS)
}

fn calculate_sse<'a>(
    it1: impl Iterator<Item = &'a f64> + Clone,
    it2: impl Iterator<Item = &'a f64> + Clone,
) -> f64 {
    let it = it1.zip(it2).enumerate().map(|(i, (n, c))| {
        // Weight exponentially per frame.
        // We want constant * (falloff^0 + falloff^1 + ... + falloff^total_frame_count) = 1.
        // We have 1 / c = falloff^0 + ... + falloff^total_frame_count = (falloff^(total_frame_count + 1) - 1) / (falloff - 1)
        let frame_index = i / SEARCH_DIMENSIONS;
        let constant = (WEIGHT_FALLOFF - 1.0)
            / (WEIGHT_FALLOFF.powi(ERROR_FRAME_HISTORY_COUNT as i32 + 1) - 1.0);
        let weight = constant * WEIGHT_FALLOFF.powi(frame_index as i32);
        weight * (n - c).powi(2)
    });

    (it.clone().sum::<f64>() / (it.count() as f64)).sqrt()
}

fn calculate_scaled_dot<'a>(
    it1: impl Iterator<Item = &'a f64> + Clone,
    it2: impl Iterator<Item = &'a f64> + Clone,
) -> f64 {
    // TODO: handle silence. maybe silence => self_dot_product and dot_product both low => dot close to 1
    fn sum<'a>(it1: impl Iterator<Item = &'a f64>, it2: impl Iterator<Item = &'a f64>) -> f64 {
        // Weight exponentially per frame.
        // We want constant * (falloff^0 + falloff^1 + ... + falloff^total_frame_count) = 1.
        // We have 1 / c = falloff^0 + ... + falloff^total_frame_count = (falloff^(total_frame_count + 1) - 1) / (falloff - 1)
        let constant = (WEIGHT_FALLOFF - 1.0)
            / (WEIGHT_FALLOFF.powi(ERROR_FRAME_HISTORY_COUNT as i32 + 1) - 1.0);
        it1.zip(it2)
            .enumerate()
            .map(|(i, (a, b))| {
                let frame_index = i / SEARCH_DIMENSIONS;
                let weight = constant * WEIGHT_FALLOFF.powi(frame_index as i32);
                a * b * weight
            })
            .sum()
    }
    let dot_product = sum(it1.clone(), it2.clone());
    let self_dot_product = sum(it2.clone(), it2);
    let normalized_dot_product = dot_product / self_dot_product;
    normalized_dot_product.ln().abs()
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
    last_prediction: Option<usize>,
}

impl PredictionManager {
    /// The range in which two timestamps are considered close enough to be the same prediction.
    const EQUAL_WINDOW: f64 = 1.0;
    const DECAY_RATIO: f64 = 21.0 / 22.0;
    const INCREMENT: f64 = 1.0;
    /// How much larger the count should be than the last prediction's count before switching.
    const JUMP_DISTANCE: f64 = 0.0;

    // TODO: make "sticky". add past best prediction, and only switch if
    // new best is some amount better than past best.

    fn add_prediction(&mut self, predicted_time: f64, current_time: f64) {
        let normalized_time = predicted_time - current_time;
        let mut found_match = false;
        for (time, count) in &mut self.data {
            // 1 second window
            if (*time - normalized_time).abs() < Self::EQUAL_WINDOW {
                // Recalculate the prediction, using the count as a weight.
                // Using count as weight helps reduce noise from predictions.
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

        let best_index = self.get_best_time_index();

        let last_prediction_value = self.last_prediction.map(|i| self.data[i]);

        // Remove everything (except the best time and last predicted time) with a counter value less than 1.
        self.data = self
            .data
            .iter()
            .enumerate()
            .filter_map(|(i, (t, c))| {
                if self.last_prediction == Some(i) || **c >= 1.0 || i == best_index {
                    Some((*t, *c))
                } else {
                    None
                }
            })
            .collect();

        // We must not have removed the last prediction, so find its new index.
        self.last_prediction = if let Some(last_prediction_value) = last_prediction_value {
            Some(
                self.data
                    .iter()
                    .position(|v| *v == last_prediction_value)
                    .unwrap(),
            )
        } else {
            None
        };
    }

    /// Returns a prediction and the closest frame that matches that prediction.
    /// Assumes that frames are always divisible by QUANTUM_SIZE.
    fn get_prediction(&mut self, current_time: f64, sample_rate: f64) -> (u64, Prediction) {
        let index = self.get_best_time_index();

        let index = if self.data[index].1
            >= self
                .last_prediction
                .map(|i| self.data[i].1)
                .unwrap_or(0.0.try_into().unwrap())
                + Self::JUMP_DISTANCE
        {
            self.last_prediction = Some(index);
            index
        } else {
            self.last_prediction.unwrap_or(0)
        };
        let best_time = self.data[index].0 + current_time;

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

    fn get_best_time_index(&self) -> usize {
        let index = self
            .data
            .iter()
            .enumerate()
            .max_by_key(|d| d.1 .1)
            .map(|d| (d.0))
            .expect("data should not be empty");
        index
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
        // TODO: check if necessary
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

            self.past_data.try_set_time(
                updated_time,
                inputs[0].channel_data(0).len(),
                scope.current_time,
            );
        }

        self.past_data.play(&mut outputs[0]);

        false
    }
}

fn log_info(data: &[Time]) {
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
}
