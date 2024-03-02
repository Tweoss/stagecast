use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::File;
use std::sync::{Arc, Mutex};

use ordered_float::NotNan;
use rand::Rng;
use rand::SeedableRng;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use web_audio_api::context::{AudioContext, AudioContextRegistration, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ChannelConfig};
use web_audio_api::render::AudioProcessor;

use fft::{
    do_fft, load_file, random_project, Time, DURATION, FFT_LEN, PENALTY, PENALTY_TIME, REPEATS,
    SEARCH_DIMENSIONS,
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
    src.connect(&context.destination());
    src.start();

    let fft = FftNode::new(
        &context,
        FftOptions {
            fft_len: FFT_LEN,
            fft_run_interval: 8,
        },
    );
    src.connect(&fft);

    // connect the node directly to the destination node (speakers)
    src.connect(&context.destination());

    // enjoy listening
    std::thread::sleep(std::time::Duration::from_secs_f32(
        DURATION * REPEATS as f32,
    ));

    let times = fft.times.lock().unwrap().to_vec();
    serde_json::to_writer(&File::create("output/data.json").unwrap(), &times).unwrap();
}

struct FftNode {
    registration: AudioContextRegistration,
    channel_config: ChannelConfig,
    times: Arc<Mutex<Vec<Time>>>,
}

struct FftOptions {
    fft_len: usize,
    /// Number of samples before running another fft
    fft_run_interval: usize,
}

impl FftNode {
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
            };
            let node = FftNode {
                registration,
                channel_config: ChannelConfig::default(),
                times,
            };

            (node, Box::new(render))
        })
    }
}

impl AudioNode for FftNode {
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
                })
            }
            self.projections
                .insert(scope.current_time.try_into().unwrap(), current_projections);
        }
        self.fft_counter += 1;
        self.fft_counter %= self.fft_run_interval;
        false
    }
}
