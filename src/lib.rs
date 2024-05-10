pub mod search;

use ordered_float::NotNan;
use rand::Rng;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use web_audio_api::context::BaseAudioContext;
use web_audio_api::AudioBuffer;

/// Samples per second
pub const ASSUMED_SAMPLE_RATE: u64 = 44100;
pub const REPEATS: usize = 1;
// TODO check best value
// 2^17 = 131072. 131072 / 44100 = 2.972154195
pub const FFT_LEN: usize = 2usize.pow(15);
pub const PROJECTION_LENGTH: usize = 400;
pub const SEARCH_DIMENSIONS: usize = 16;
/// How close a frame should be to consider it as effectively current time.
pub const IGNORE_FRAME: u64 = ASSUMED_SAMPLE_RATE;
pub const PENALTY: f64 = 10.0;
/// Assumed render quantum size
pub const QUANTUM_SIZE: u64 = 128;

/// How many samples from the start and ends of the fft input
/// to smooth down. Reduces noise in fft output.
const SMOOTHING_COUNT: usize = QUANTUM_SIZE as usize * 5;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Time {
    pub real: f64,
    pub predicted: f64,
    pub projection: Vec<NotNan<f32>>,
    pub error: f64,
    pub managed_prediction: f64,
    pub dot_error: f64,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct FftSample {
    pub real: f64,
    pub magnitudes: Vec<f32>,
}

pub fn do_fft(planner: &mut RealFftPlanner<f32>, input: &mut [f32], output: &mut [Complex<f32>]) {
    let length = input.len();

    // scale down ends.
    input
        .iter_mut()
        .take(SMOOTHING_COUNT)
        .enumerate()
        .for_each(|(i, v)| *v *= (i as f32 / SMOOTHING_COUNT as f32).powi(2));
    input
        .iter_mut()
        .rev()
        .take(SMOOTHING_COUNT)
        .enumerate()
        .for_each(|(i, v)| *v *= (i as f32 / SMOOTHING_COUNT as f32).powi(2));

    // create a FFT
    let r2c = planner.plan_fft_forward(length);

    // forward transform the signal
    r2c.process(input, output).unwrap();

    output.iter_mut().for_each(|o| *o /= (length as f32).sqrt());
}

pub struct RandomSubspace<const OUTPUT_DIMS: usize, const INPUT_DIMS: usize> {
    pub values: Box<[[f32; INPUT_DIMS]; OUTPUT_DIMS]>,
}

// 2^15 = 32768
// 44100 / 2^15 = 1.3458252
// 44100/(440*32) = 3.13210227
// 44100/(440)*4 = 400.90909091
impl<const OUTPUT_DIMS: usize, const INPUT_DIMS: usize> RandomSubspace<OUTPUT_DIMS, INPUT_DIMS> {
    pub fn new(rng: &mut impl Rng) -> Self {
        let mut output = Box::new([[0.0; INPUT_DIMS]; OUTPUT_DIMS]);
        for row in output.iter_mut() {
            row.fill_with(|| rng.gen_range(-1.0..1.0))
        }
        Self { values: output }
    }

    pub fn random_project(&self, input: &[Complex<f32>]) -> Vec<f64> {
        assert!(input.len() >= INPUT_DIMS);

        self.values
            .iter()
            .map(|r| {
                r.iter()
                    .zip(input.iter().take(INPUT_DIMS))
                    .map(|(v, c)| (*v * c.norm_sqr().sqrt()) as f64)
                    .sum()
            })
            .collect::<Vec<_>>()
    }
}

pub fn generate_varying_sine(context: &impl BaseAudioContext, duration: f32) -> AudioBuffer {
    use std::f32::consts::PI;

    let length = (context.sample_rate() * duration) as usize;
    let sample_rate = context.sample_rate();
    let mut buffer = context.create_buffer(1, length, sample_rate);

    // Fill buffer with a varying sine wave
    let mut sine = vec![];

    // Vary pitch from A, to A
    let (start, end) = (220.0, 440.0);
    let lerp = |proportion: f32| -> f32 { (1.0 - proportion) * start + proportion * end };

    for i in 0..length {
        let frequency = lerp(i as f32 / length as f32);
        let phase = i as f32 / sample_rate * 2. * PI * frequency;
        // Add some harmonics.
        sine.push(phase.sin() + (phase * 2.).sin() + (phase * 1.5).sin());
    }

    buffer.copy_to_channel(&sine, 0);

    buffer
}

pub fn load_file(context: &impl BaseAudioContext, path: &str) -> AudioBuffer {
    // for background music, read from local file
    let file = std::fs::File::open(path).unwrap();
    context.decode_audio_data_sync(file).unwrap()
}
