use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use web_audio_api::context::BaseAudioContext;
use web_audio_api::AudioBuffer;

pub const SEARCH_DIMENSIONS: usize = 2usize.pow(4);
// pub const DURATION: f32 = 10.;
// Turkish March
// pub const DURATION: f32 = 2. * 60. + 40.;
// La Dispute
pub const DURATION: f32 = 13. * 60.;
pub const REPEATS: usize = 1;
// TODO check best value
// 2^17 = 131072. 131072 / 44100 = 2.972154195
pub const FFT_LEN: usize = 2usize.pow(15);
/// How close a frame should be to apply a penalty.
// pub const PENALTY_FRAME: u64 = 44100 * 13 * 60 / 4;
pub const PENALTY_FRAME: u64 = 44100 * 8;
pub const PENALTY: f64 = 10.0;
/// Assumed render quantum size
pub const QUANTUM_SIZE: u64 = 128;
/// How close a new predicted frame must be to the last predicted time to get a bonus.
pub const BONUS_FRAME: u64 = QUANTUM_SIZE * 10;
/// The ratio of artificial decrease in error.
pub const BONUS: f64 = 2.0;

/// How many samples from the start and ends of the fft input
/// to smooth down. Reduces noise in fft output.
const SMOOTHING_COUNT: usize = QUANTUM_SIZE as usize * 5;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Time {
    pub real: f64,
    pub predicted: f64,
    pub projection: Vec<f64>,
    pub error: f64,
    pub managed_prediction: f64,
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
}

pub struct RandomVector {
    pub points: Vec<f32>,
}

pub fn random_project(input: &[Complex<f32>], projection: &RandomVector) -> f32 {
    assert!(input.len() == projection.points.len());
    input
        .iter()
        // atan is a very expensive operation so we just use magnitude not phase shift
        .map(|v| v.norm_sqr().sqrt())
        .zip(projection.points.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>()
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

pub fn load_file(context: &impl BaseAudioContext, duration: f32, path: String) -> AudioBuffer {
    // for background music, read from local file
    let file = std::fs::File::open(path).unwrap();
    let buffer = context.decode_audio_data_sync(file).unwrap();
    let mut output = context.create_buffer(
        1,
        (duration * context.sample_rate()) as usize,
        context.sample_rate(),
    );
    output.copy_to_channel(buffer.get_channel_data(0), 0);
    output
}
