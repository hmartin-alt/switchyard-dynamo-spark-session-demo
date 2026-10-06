# Prefill batching

Batching groups encoder work across requests. Each request still goes independently to its selected Dynamo model with its full input.

## Source files

These overlays extend the local Switchyard integration:

| File | Destination in Switchyard |
| --- | --- |
| `algorithm.rs` | `crates/prefill-router/src/algorithm.rs` |
| `microbatch.rs` | `crates/prefill-router/src/microbatch.rs` |
| `transformers_forward.py` | `crates/prefill-router/python/transformers_forward.py` |
| `algorithm-tests.rs` | `crates/prefill-router/tests/unit/algorithm.rs` |

Place `switchyard_batch.py` on the serving Python import path. The [runtime build script](../../scripts/shopping-mmlu/restore-colocated-runtime.sh) installs the serving overlays and rebuilds the binary using Rust 1.96.1. Copy the test overlay separately before running the Rust tests.

```bash
cargo test --release --locked -p prefill-router
cargo build --release --locked -p switchyard-server --features prefill-router
```

## Serving settings

`manifests/router/start-colocated.sh` enables batches of up to 16 requests with a 2 ms collection window. Queueing and inference take additional time.

Requests near the routing threshold are rescored individually to reduce batch-rounding differences. This safeguard is not a universal numerical guarantee: recheck prediction parity after changing the checkpoint, runtime, or batching settings.

Keep the launcher's tolerance aligned with `colocated-routes.toml`. To disable batching, omit `SWITCHYARD_PREFILL_MICROBATCH`.
