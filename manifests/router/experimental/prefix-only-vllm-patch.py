"""Experimental vLLM 1.4.0 policy: cache only the first 32 shared tokens.

Run before dynamo.vllm inside each ephemeral worker container. This is not a
general production cache policy. It reproduces the controlled shared-prefix
comparison while preventing complete repeated questions from hitting cache.
"""

from importlib.util import find_spec
from pathlib import Path


TARGET = "max_cache_hit_length = request.num_tokens - 1"
REPLACEMENT = (
    "max_cache_hit_length = min(request.num_tokens - 1, 32)"
    "  # Switchyard shared-prefix-only pilot"
)

spec = find_spec("vllm.v1.core.kv_cache_manager")
if spec is None or spec.origin is None:
    raise RuntimeError("vLLM v1 KV cache manager not found; refusing to run")
path = Path(spec.origin)
source = path.read_text()
if source.count(REPLACEMENT) == 1 and TARGET not in source:
    print(f"verified shared-prefix-only cache cap in {path}", flush=True)
elif source.count(TARGET) == 1 and REPLACEMENT not in source:
    path.write_text(source.replace(TARGET, REPLACEMENT, 1))
    verified = path.read_text()
    if verified.count(REPLACEMENT) != 1 or TARGET in verified:
        raise RuntimeError("vLLM cache patch verification failed")
    print(f"installed shared-prefix-only cache cap in {path}", flush=True)
else:
    raise RuntimeError("unexpected vLLM KV cache manager version; refusing to run")
