# Running

```
cargo run --release --bin predict [PATH_TO_OUTPUT_FOLDER] [PATH_TO_SAMPLE_FILE | wave | microphone]
```

After running `predict`, run
```
cargo run --release --bin graph [PATH_TO_INPUT_FOLDER]
```
to get a graph of the predicted times and frequencies.

# Rough Description

Uses the Fast Fourier Transform on recent past history, then random projection
of the results of the FFT to collapse to a few dimensions. 

Searches for similar past quantum frames, then seeks to the recording from 
those times and plays.

