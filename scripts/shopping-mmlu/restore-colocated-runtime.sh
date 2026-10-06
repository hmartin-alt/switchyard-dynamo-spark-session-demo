#!/usr/bin/env bash
# Run in the colocated-builder pod after copying source and patches to /runtime.
set -euo pipefail
cd /runtime
apt-get update
apt-get install -y build-essential pkg-config libssl-dev curl patch
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/install-rust.sh
sh /tmp/install-rust.sh -y --profile minimal --default-toolchain 1.96.1
export PATH=/home/dynamo/.cargo/bin:/root/.cargo/bin:$PATH
python3 -m pip install --target /runtime/python --no-deps scikit-learn joblib threadpoolctl narwhals scipy
cp patches/batching/algorithm.rs switchyard-src/crates/prefill-router/src/algorithm.rs
cp patches/batching/microbatch.rs switchyard-src/crates/prefill-router/src/microbatch.rs
cp patches/batching/transformers_forward.py switchyard-src/crates/prefill-router/python/transformers_forward.py
cp patches/batching/switchyard_batch.py python/switchyard_batch.py
cd /runtime/llm-router-src
patch --forward -p1 < /runtime/patches/prefill-inference-overhead.patch
patch --forward -p1 < /runtime/patches/prefill-unused-layers.patch
cd /runtime/switchyard-src
PYO3_PYTHON=python3 cargo build --release --locked -p switchyard-server --features prefill-router
cp target/release/switchyard-server /runtime/switchyard-server-profiled
echo 'Router runtime restored.'
