use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::File;
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
    do_fft, load_file, random_project, Time, BONUS, BONUS_TIME, DURATION, FFT_LEN, PENALTY,
    PENALTY_TIME, REPEATS, SEARCH_DIMENSIONS,
};

fn main() {
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

    let fft = FftNode::new(
        &context,
        FftOptions {
            fft_len: FFT_LEN,
            fft_run_interval: 8,
        },
    );
    let (fft, prediction_rx) = fft.consume_receiver();
    src.connect(&fft);

    let left_panner = context.create_stereo_panner();
    left_panner.connect(&context.destination());
    left_panner.pan().set_value(-1.);
    src.connect(&left_panner);

    let context = Arc::new(context);
    let playback = PlaybackNode::new(context.clone(), PlaybackOptions { prediction_rx });
    src.connect(&playback);

    let right_panner = context.create_stereo_panner();
    right_panner.connect(&context.destination());
    right_panner.pan().set_value(1.);
    playback.connect(&right_panner);

    src.start();
    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs_f32(
        DURATION * REPEATS as f32,
    ));

    let times = fft.times.lock().unwrap().to_vec();
    serde_json::to_writer(&File::create("output/data.json").unwrap(), &times).unwrap();
}

struct FftNode<PredictionType> {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    times: Arc<Mutex<Vec<Time>>>,
    prediction: PredictionType,
}

struct FftOptions {
    fft_len: usize,
    /// Number of samples before running another fft
    fft_run_interval: usize,
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
                planner: real_planner,
                past_input,
                data: vec![Complex::new(0.0, 0.0); options.fft_len / 2 + 1],
                fft_run_interval: options.fft_run_interval,
                fft_counter: 0,
                random_vector: (0..SEARCH_DIMENSIONS)
                    .map(|_| gen_random_vector())
                    .collect(),
                btree: (0..SEARCH_DIMENSIONS).map(|_| BTreeMap::new()).collect(),
                projections: HashMap::new(),
                times: times.clone(),
                last_predicted_time: (0.0, tx),
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
    past_input: VecDeque<f32>,
    data: Vec<Complex<f32>>,
    planner: RealFftPlanner<f32>,
    fft_run_interval: usize,
    fft_counter: usize,
    random_vector: Vec<Vec<f32>>,
    /// Map from random projection to timestamp
    btree: Vec<BTreeMap<NotNan<f32>, f64>>,
    /// Map from timestamp to projections.
    projections: HashMap<NotNan<f64>, Vec<f64>>,
    times: Arc<Mutex<Vec<Time>>>,
    last_predicted_time: (f64, Sender<Prediction>),
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

        if self.fft_counter == 0 {
            // TODO try scratch space, storing Fft after planning
            do_fft(
                &mut self.planner,
                self.past_input.clone().make_contiguous(),
                &mut self.data,
            );

            // TODO try benchmarking this (maybe put in a for loop so flamegraph can see)
            let current_projections: Vec<_> = self
                .random_vector
                .iter()
                .map(|v| random_project(&self.data, v) as f64)
                .collect();

            // Find the best guess among neighbors.
            if let Some(best_guess) = current_projections
                .iter()
                .zip(self.btree.iter_mut())
                .flat_map(|(p, b)| {
                    let projection = NotNan::new(*p as f32).unwrap();
                    let neighbors = b
                        .range(..projection)
                        .last()
                        .map(|(_, t)| *t)
                        .into_iter()
                        .chain(b.range(projection..).next().map(|(_, t)| *t));
                    // Note: this is a side effect, updating btrees.
                    // Thus, this line *needs* to be after searching for neighbors.
                    b.insert(projection, scope.current_time);
                    neighbors
                })
                .map(|n_time| {
                    // Using sum squared error over all projections.
                    let neighbor_projections = self
                        .projections
                        .get(&NotNan::new(n_time).unwrap())
                        .expect("past time should have entry");
                    (
                        NotNan::new(
                            (neighbor_projections
                                .iter()
                                .zip(current_projections.iter())
                                .map(|(n, c)| (n - c).powi(2))
                                .sum::<f64>()
                                / current_projections.len() as f64)
                                .sqrt(),
                        )
                        .unwrap()
                            + if (scope.current_time - n_time) < PENALTY_TIME {
                                // Add some error for nearby times
                                PENALTY
                            } else {
                                0.0
                            }
                            - if (scope.current_time - self.last_predicted_time.0) < BONUS_TIME {
                                // Decrease error for times near last predicted time
                                BONUS
                            } else {
                                0.0
                            },
                        n_time,
                    )
                })
                .min_by_key(|(error, _)| *error)
            {
                self.times.lock().unwrap().push(Time {
                    real: scope.current_time,
                    predicted: best_guess.1,
                    projection: current_projections.clone(),
                });
                self.last_predicted_time.0 = best_guess.1;
                let _ = self.last_predicted_time.1.send(Prediction {
                    predicted_time: best_guess.1,
                    time_of_prediction: scope.current_time,
                });
            }
            self.projections
                .insert(scope.current_time.try_into().unwrap(), current_projections);
        }
        self.fft_counter += 1;
        self.fft_counter %= self.fft_run_interval;
        false
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
