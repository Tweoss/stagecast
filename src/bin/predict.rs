use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::File;
use std::ops::Bound::Excluded;
use std::ops::Bound::Unbounded;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use ordered_float::NotNan;
use rand::Rng;
use rand::SeedableRng;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use web_audio_api::context::{AudioContext, AudioContextRegistration, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ChannelConfig};
use web_audio_api::render::{AudioProcessor, AudioRenderQuantum};
use web_audio_api::AudioBuffer;

use fft::{
    do_fft, load_file, random_project, Time, BONUS, BONUS_FRAME, DURATION, FFT_LEN, PENALTY,
    PENALTY_FRAME, REPEATS, SEARCH_DIMENSIONS,
};

/// How many previous frames of projections to look at when calculating sum
/// squared error.
const ERROR_FRAME_HISTORY_COUNT: u64 = 1;
/// Neighbor count per side per projection
const NEIGHBOR_COUNT: usize = 1;

fn main() {
    let output_file = std::env::args()
        .nth(1)
        .expect("pass the output json path as an argument");

    // let context = OfflineAudioContext::new(1, 44100 * DURATION as usize, 44100.0);
    let context = AudioContext::default();

    // let mic = media_devices::get_user_media_sync(MediaStreamConstraints::Audio);
    // // register as media element in the audio context
    // let src = context.create_media_stream_source(&mic);

    // play a buffer in a loop
    let mut src = context.create_buffer_source();
    src.set_buffer(
        // load a file
        load_file(
            &context,
            DURATION,
            // "/Users/francischua/Downloads/sample2.ogg".to_owned(),
            "/Users/francischua/Downloads/ladispute_amp.ogg".to_owned(),
        ),
        // or generate a sample (escalating chord stack)
        // generate_varying_sine(&context, DURATION),
    );
    src.set_loop(true);

    let fft = FftNode::new(&context, FftOptions { fft_len: FFT_LEN });
    let (fft, prediction_rx) = fft.consume_receiver();
    src.connect(&fft);

    let left_panner = context.create_stereo_panner();
    left_panner.connect(&context.destination());
    left_panner.pan().set_value(-1.);

    let context = Arc::new(context);
    let playback = PlaybackNode::new(context.clone(), PlaybackOptions { prediction_rx });
    src.connect(&playback);

    let right_panner = context.create_stereo_panner();
    right_panner.connect(&context.destination());
    right_panner.pan().set_value(1.);
    src.connect(&right_panner);
    playback.connect(&left_panner);

    src.start();
    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs_f32(
        DURATION * REPEATS as f32,
    ));

    let times = fft.times.lock().unwrap().to_vec();
    serde_json::to_writer(&File::create(output_file).unwrap(), &times).unwrap();
}

struct FftNode<PredictionType> {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    times: Arc<Mutex<Vec<Time>>>,
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
            // Deterministic for reproducibility.
            let mut rng = rand::rngs::SmallRng::seed_from_u64(0xDEADBEEF);
            let mut gen_random_vector = || {
                let mut vec = (0..(options.fft_len))
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Vec<f32>>();
                // Normalize to unit.
                let magnitude = vec.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
                for el in &mut vec {
                    *el /= magnitude
                }
                vec
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
                last_prediction: (0, tx),
            };
            let node = FftNode {
                registration,
                channel_config: ChannelConfig::default(),
                times,
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
    random_vector: Vec<Vec<f32>>,
    frame_data: FrameData,
    times: Arc<Mutex<Vec<Time>>>,
    last_prediction: (u64, Sender<Prediction>),
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

        // TODO try benchmarking this (maybe put in a for loop so flamegraph can see)
        // this is indeed some 68% of runtime. perhaps reduce dimensions
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
                    ))
                    .unwrap(),
                    n_frame,
                )
            })
            .min_by_key(|(error, _)| *error)
        {
            if *error <= 20.0 {
                let predicted_time = best_frame as f64 / self.sample_rate;
                self.times.lock().unwrap().push(Time {
                    real: scope.current_time,
                    predicted: predicted_time,
                    projection: current_projections.clone(),
                    error: *error,
                });
                self.last_prediction.0 = best_frame;
                let _ = self.last_prediction.1.send(Prediction {
                    predicted_time,
                    time_of_prediction: scope.current_time,
                });
            }
        }

        false
    }
}

fn calculate_error(
    projections: &HashMap<u64, (usize, Vec<f64>)>,
    frames: &[u64],
    neighbor_frame: u64,
    current_frame: u64,
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

    // Since we don't know (may change across web_audio_api versions) the quantum length,
    // we just search backwards until we get enough frames (or run out of samples)
    let it = frames[..=neighbor_index]
        .iter()
        .rev()
        .zip(frames[..=current_index].iter().rev())
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
    (it.clone().sum::<f64>() / (it.count() as f64)).sqrt()
    // let frame_delta = current_frame.abs_diff(neighbor_frame);

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

    // TODO handle last predicted time
    // if (current_frame - self.last_predicted_time.0) < BONUS_FRAME {
    //     // Decrease error for times near last predicted time
    //     error -= BONUS
    // }

    // error
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

struct Prediction {
    predicted_time: f64,
    time_of_prediction: f64,
}

struct ResizingBuffer {
    buffer: AudioBuffer,
    used: usize,
    start_time: f64,
    playback_index: usize,
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
    fn try_set_time(&mut self, time: f64, sample_len: usize) -> bool {
        let sample_duration = sample_len as f64 / self.buffer.sample_rate() as f64;
        let time_tolerance = 100. * sample_duration;
        let new_time = if (self.get_time() - time).abs() < time_tolerance {
            return false;
        } else {
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
            .zip(self.buffer.get_channel_data(0)[self.playback_index..].iter())
            .map(|(o, b)| *o = *b)
            .count();
        self.playback_index += written_count;
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
            self.past_data
                .try_set_time(updated_time, inputs[0].channel_data(0).len());
        }
        // // Only play if much in the past.
        // if scope.current_time - self.past_data.get_time() > DURATION as f64 {
        //     self.past_data.play(&mut outputs[0]);
        // }
        self.past_data.play(&mut outputs[0]);

        false
    }
}
