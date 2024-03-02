Run this after editing `src/bin/predict.rs` to point to a suitable audio file.

```
cargo run --release --bin predict
```

After running `predict`, run
```
cargo run --release --bin graph
```
to get a graph of the predicted times.
