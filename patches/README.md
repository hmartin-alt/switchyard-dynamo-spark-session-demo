# Local Switchyard integration

`switchyard-prefill-local.patch` captures the local tolerance, toolkit-checkpoint,
and CUDA-device fixes against Switchyard commit `a601a9a3f9db149a1ad430fa43b1463a170c8a82`.
These are local changes, not a claim about upstream support.

Apply to that checkout with `git apply`, then rebuild:

```bash
cargo build --release --locked -p switchyard-server --features prefill-router
```

The Linux experiment uses the same patched source as the Mac build. Install the
model-router toolkit into the Python environment embedded by the server.
