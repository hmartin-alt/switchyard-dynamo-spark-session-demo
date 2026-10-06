"""Batched toolkit scoring; same checkpoint features, no outcome-based changes."""
import json
import os
import time
import numpy as np
from model_router_toolkit.prefill.transforms import build_trunk_features
from model_router_toolkit.prefill.trunk import predict_proba


def _sync_cuda(scorer):
    torch = getattr(scorer, "_torch", None)
    if torch is None:
        try:
            import torch
        except ImportError:
            return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def score_batch(scorer, prompts, batch_size, max_length):
    load_started = time.perf_counter()
    scorer._ensure_loaded()
    extractor = scorer._extractor
    extractor._ensure_loaded()
    _sync_cuda(scorer)
    load_ms = (time.perf_counter() - load_started) * 1000
    checkpoint = scorer._ckpt
    # Existing pooling reads tokens [:seq_len]. Make that correct for padding.
    extractor._tokenizer.padding_side = "right"
    cache, results = {}, {}
    encoder_ms = 0.0
    original_forward = extractor._forward_for_features

    def timed_forward(*args, **kwargs):
        nonlocal encoder_ms
        _sync_cuda(scorer)
        started = time.perf_counter()
        output = original_forward(*args, **kwargs)
        _sync_cuda(scorer)
        encoder_ms += (time.perf_counter() - started) * 1000
        return output

    extractor._forward_for_features = timed_forward
    extraction_started = time.perf_counter()
    try:
        for name in scorer.model_names:
            transform = checkpoint["transforms"][name]
            if transform["encoder"] != extractor._hf_path:
                raise ValueError("Batched scorer requires a shared encoder")
            template = transform.get("chat_template_kwargs", {})
            key = (transform["encoder"], json.dumps(template, sort_keys=True))
            if key not in cache:
                cache[key] = extractor.extract_batch(
                    prompts, chat_template_kwargs=template,
                    extract_layers=scorer._needed_layers(),
                    pooling_modes=scorer._needed_pooling_modes(),
                    batch_size=batch_size, max_length=max_length, show_progress=False)
            results[name] = cache[key]
    finally:
        extractor._forward_for_features = original_forward
    _sync_cuda(scorer)
    extraction_ms = (time.perf_counter() - extraction_started) * 1000
    head_started = time.perf_counter()
    features = build_trunk_features(results, checkpoint["transforms"], scorer.model_names,
        checkpoint.get("trunk_config", {}).get("feature_layout", "per_target"))
    scores = predict_proba(scorer._trunk_nets, features, device=scorer._device)
    _sync_cuda(scorer)
    head_ms = (time.perf_counter() - head_started) * 1000
    return scores, {
        "load_ms": load_ms,
        "extract_ms": extraction_ms,
        "encoder_forward_ms": encoder_ms,
        "tokenize_pool_ms": max(0.0, extraction_ms - encoder_ms),
        "routing_head_ms": head_ms,
    }


def forward(scorer, prompts, batch_size, max_length):
    total_started = time.perf_counter()
    if len(prompts) == 1:
        scores = np.asarray([scorer.score(prompts[0]).confidences], dtype=np.float32)
        _sync_cuda(scorer)
        print(json.dumps({"event":"prefill_profile", "n":1, "scalar_path":True,
                          "total_ms":(time.perf_counter()-total_started)*1000}), flush=True)
        return scores
    scores, timings = score_batch(scorer, prompts, batch_size, max_length)
    # Batch-shape BF16 rounding can change choices close to the fixed policy
    # boundary. Re-score those requests with the original scalar path.
    tolerance = float(os.environ["SWITCHYARD_PREFILL_TOLERANCE"])
    guard = float(os.environ.get("SWITCHYARD_PREFILL_NUMERIC_GUARD", "0.06"))
    if not (0 <= tolerance <= 1 and 0 <= guard <= 1):
        raise ValueError("Tolerance and numerical guard must be in [0, 1]")
    threshold = scores.max(axis=1) - np.float32(tolerance)
    borderline = np.any(abs(scores-threshold[:, None]) <= guard, axis=1)
    recheck_started = time.perf_counter()
    for i in np.flatnonzero(borderline):
        scores[i] = scorer.score(prompts[i]).confidences
    _sync_cuda(scorer)
    timings["scalar_recheck_ms"] = (time.perf_counter() - recheck_started) * 1000
    timings["total_ms"] = (time.perf_counter() - total_started) * 1000
    print(json.dumps({"event":"prefill_profile", "n":len(prompts),
                      "scalar_rechecks":int(borderline.sum()), **timings}), flush=True)
    return scores
