# Build the local Switchyard runtime

This setup uses local changes for toolkit-checkpoint loading, tolerance selection, GPU execution, and encoder performance. Building the runtime compiles those changes into the serving program; it does not retrain the router.

## Required changes

- `switchyard-prefill-local.patch` — Switchyard checkpoint, tolerance, and device integration.
- `prefill-inference-overhead.patch` — toolkit inference optimizations.
- `prefill-unused-layers.patch` — skip unused encoder layers.
- [Batching overlays](batching/README.md) — score multiple requests together.

The Switchyard integration patch targets commit `a601a9a3f9db149a1ad430fa43b1463a170c8a82`. Keep this source version when rebuilding; patches may not apply to another version.

## Build

1. Place a **fresh, unpatched** matching Switchyard checkout at `/runtime/switchyard-src`, compatible toolkit source at `/runtime/llm-router-src`, and this patch directory at `/runtime/patches` in the builder pod.
2. Place `scripts/shopping-mmlu/restore-colocated-runtime.sh` at `/runtime/restore-colocated-runtime.sh` and run it **once**. It checks and applies the Switchyard integration patch, applies toolkit optimizations and batching overlays, installs build dependencies, and builds `switchyard-server-profiled`.

The script expects prepared source trees; it does not download them. Use a fresh toolkit copy when applying its patches. The serving Python environment must include the model-router toolkit and its dependencies. The source and checkpoint are not distributed by this repository.

See the [deployment guide](../docs/shopping-demo.md#deployment-files) for the runtime mounts and launcher.
