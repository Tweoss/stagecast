use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use web_audio_api::context::BaseAudioContext;
use web_audio_api::AudioBuffer;

pub const SEARCH_DIMENSIONS: usize = 2usize.pow(5);
// const DURATION: f32 = 10.;
pub const DURATION: f32 = 13. * 60. + 4.;
pub const REPEATS: usize = 1;
// TODO check best value
// 2^17 = 131072. 131072 / 44100 = 2.972154195
pub const FFT_LEN: usize = 2usize.pow(17);
pub const PENALTY_TIME: f64 = 2.0;
pub const PENALTY: f64 = 0.5;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Time {
    pub real: f64,
    pub predicted: f64,
    pub projection: Vec<f64>,
}

pub fn do_fft(planner: &mut RealFftPlanner<f32>, input: &mut [f32], output: &mut [Complex<f32>]) {
    let length = input.len();

    // create a FFT
    let r2c = planner.plan_fft_forward(length);

    // forward transform the signal
    r2c.process(input, output).unwrap();
}

pub fn random_project(input: &[Complex<f32>], projection: &[f32]) -> f32 {
    // TODO since ignoring angle, reduce projection vector size
    assert!(input.len() == projection.len() / 2 + 1);
    input
        .iter()
        .map(|v| (v.norm_sqr().sqrt()))
        // atan is a very expensive operation
        // .flat_map(|v| [v.norm_sqr().sqrt(), v.atan().re])
        .zip(projection.iter())
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

    // Test with constant pitch.
    // let (start, end) = (440.0, 440.0);

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
