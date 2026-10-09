#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/router-runtime/python:/router-runtime/llm-router-src
export HF_HOME=/hf-cache HF_HUB_CACHE=/hf-cache/hub DUMMY_KEY=unused
export PREFILL_RETAIN_CUDA_CACHE=1 PREFILL_LAST_LOGIT_ONLY=1 PREFILL_SKIP_UNUSED_LAYERS=1
export SWITCHYARD_PREFILL_MICROBATCH=1 SWITCHYARD_PREFILL_BATCH_SIZE=16 SWITCHYARD_PREFILL_BATCH_WAIT_MS=2
export SWITCHYARD_PREFILL_NUMERIC_GUARD=0.0024222002754211427
export SWITCHYARD_PREFILL_TOLERANCE=0.1760250329971314

case "${SWITCHYARD_PREFIX_CACHE_MODE:-disabled}" in
  disabled) cache_flag=--no-enable-prefix-caching ;;
  shared-prefix-only)
    python3 /router-runtime/experimental/prefix-only-vllm-patch.py
    cache_flag=--enable-prefix-caching ;;
  *) echo "Unsupported SWITCHYARD_PREFIX_CACHE_MODE" >&2; exit 2 ;;
esac

python3 -m dynamo.vllm \
  --model /hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --served-model-name efficient --max-model-len 8192 --gpu-memory-utilization 0.72 \
  "$cache_flag" &
worker_pid=$!
cleanup() {
  kill "$worker_pid" "${router_pid:-}" 2>/dev/null || true
  wait "$worker_pid" "${router_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Load vLLM before the colocated encoder claims GPU memory.
until curl -fsS http://127.0.0.1:9090/live >/dev/null; do
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    wait "$worker_pid"
    exit $?
  fi
  sleep 2
done

/router-runtime/switchyard-server-profiled \
  --config /router-runtime/routes.toml --port 4000 \
  --routing-log-file /router-runtime/routing-production-efficient.jsonl &
router_pid=$!
while kill -0 "$worker_pid" 2>/dev/null && kill -0 "$router_pid" 2>/dev/null; do
  sleep 2
done
exit 1
