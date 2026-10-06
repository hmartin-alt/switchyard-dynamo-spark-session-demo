# Experimental native prefill microbatching

These source overlays extend the existing local Switchyard toolkit-checkpoint
adapter; they are not an upstream feature claim. They were built against the
existing `/workspace/switchyard-src` checkout with Rust 1.96.1. Preserve the
original files and binary before applying; do not copy onto unrelated versions.

Mappings within that checkout:

- `algorithm.rs` → `crates/prefill-router/src/algorithm.rs`
- `microbatch.rs` → `crates/prefill-router/src/microbatch.rs`
- `transformers_forward.py` → `crates/prefill-router/python/transformers_forward.py`
- `algorithm-tests.rs` → `crates/prefill-router/tests/unit/algorithm.rs`

Place `switchyard_batch.py` on the serving Python import path. The live experiment
uses `/router-runtime/llm-router-src`, retaining the cache/logit/layer
optimizations. The Python adapter is embedded in the Rust binary and requires a
rebuild. The helper module is imported at runtime.

```bash
cargo test --release --locked -p prefill-router
cargo build --release --locked -p switchyard-server --features prefill-router
```

Enable with `manifests/router/start-colocated.sh`. Current settings:

- queue capacity: 128; overload returns an error, never a guessed route
- maximum request batch: 16
- batch collection deadline: 2 ms (queue/inference time is additional)
- tolerance: 0.1760250329971314, matching the native route configuration
- numerical-boundary guard: 0.0024222002754211427; borderline scores get a scalar recheck

Only scoring is batched; each original request and its full messages continue to
its selected Dynamo backend independently. Result ordering is preserved. Dropped
requests are skipped before inference when possible; errors are delivered to the
affected requests, and the worker can process later requests.

The numerical guard is a conservative heuristic, not a universal floating-point
error bound. It is independent of outcome labels and does not alter the tolerance.
Validation parity must be rechecked for new checkpoints, encoders, runtimes, or
input distributions. Batch scores can differ even when routing choices match.

Flags default off in the adapter and are enabled by the colocated launcher.
The serial path remains available by omitting `SWITCHYARD_PREFILL_MICROBATCH`.
Changing batching settings requires new parity checks; do not assume identical
predictions on a different checkpoint or runtime.
