# Current shopping routing experiment

## Workload and data boundaries

The dataset contains real Shopping MMLU questions and their original multiple-choice answers. Task-level quotas create an intentionally easy-heavy shopping workload; these results are not an estimate for all Amazon Ads traffic or the full Shopping MMLU benchmark.

- Training: 3,500 questions (2,850 easy, 650 hard).
- Validation: 500 questions (400 easy, 100 hard), used to select the policy.
- Test: 500 questions (400 easy, 100 hard), evaluated after freezing the policy.
- Easy tasks: direct attributes, categories, compatibility, and commonsense.
- Hard tasks: product numerical reasoning and unit conversion.

## Frozen policy

The score is `lambda * predicted_correctness - (1-lambda) * normalized_token_cost`. Fixed model costs use training-token medians and the saved per-token price snapshot, normalized by capable-model cost.

Validation selected lambda `0.816052255920879`, equivalent to native two-model tolerance `0.1760250329971314`. This permits the efficient target when its predicted correctness is within that tolerance of the best target; it does not guarantee that every selected request loses no more than that much actual accuracy.

## Final held-out results

| Metric | Efficient | Capable | Routed |
| --- | ---: | ---: | ---: |
| Accuracy | 83.2% | 95.2% | 93.6% |
| Mean end-to-end latency | 0.197s | 0.404s | 0.350s |
| Token-priced dollars per 1,000 requests | 0.002399 | 0.010954 | 0.004396 |

Routing selected efficient for 386/500 requests and capable for 114/500. The same-mix random baseline has expected accuracy 85.936%. The token-priced cost reduction is 59.9%, not a demonstrated reduction in the fixed EC2 fleet bill.

Prices in `configs/prefill-router/shopping-costs-token-proxy.json` are a frozen experiment snapshot, not a live price quote. Router compute, idle capacity, storage, and networking are excluded from that view. GPU colocation reduces routing overhead but does not make routing free.

## Reproduction prerequisites and workflow

Use Python 3.10+, the model-router toolkit's prefill/training dependencies, OpenAI's Python client, and Matplotlib. Live serving requires EKS with the Dynamo operator, GPU worker capacity, and the `hf-token-secret` Kubernetes secret in namespace `dynamo`. Never commit that secret.

The retained deployment files are:

- `manifests/dynamo/prefetch-job.yaml`: populate model caches.
- `manifests/router/colocated-builder.yaml`: build the patched runtime with `scripts/shopping-mmlu/restore-colocated-runtime.sh`.
- `manifests/router/train-large.yaml`: GPU training worker.
- `manifests/dynamo/dgd-efficient-colocated.yaml`: E4B plus encoder/router.
- `manifests/dynamo/dgd-capable.yaml`: 31B workers.
- `manifests/router/colocated-service.yaml`: router service.
- `manifests/router/colocated-routes.toml` and `start-colocated.sh`: frozen serving policy and launcher.

These manifests use node-local hostPath storage. Restore the runtime and model snapshots on every scheduled node, or pin pods to the prepared node. Verify the snapshot paths before starting workers. The capable manifest requests two replicas; size or adjust that intentionally for available GPUs. Apply resources explicitly in namespace `dynamo`.

The build script expects the already locally patched Switchyard source and toolkit source in the builder's `/runtime`; it is not a source downloader. See both patch READMEs before building.

1. Obtain the official Shopping MMLU archive under its upstream terms. `tiered-large.py` verifies its expected hash. Preserve the prior pilot inputs needed to reproduce the original exclusion list; a fresh dataset build is not automatically identical to this run.
2. Build the dataset with `scripts/shopping-mmlu/tiered-large.py --archive <data.zip> --output <new-directory>`. Inspect its manifest and frozen split hashes.
3. Collect both models' development predictions using `tiered-large-evaluate.py`, then prepare router labels with `tiered-large-router.py`. These experiment scripts contain fixed artifact paths: inspect their defaults before running, and do not overwrite historical results.
4. Train the router on training data only using `configs/prefill-router/shopping-pool.yaml` and the toolkit's training command. Save it under `checkpoints/shopping-tiered-real-large-v1/`. Before sweeping, save a `checkpoint.json` in the router artifact directory with a `checkpoint_sha256` field matching that file. Sweep token-priced policy on validation, then freeze it with `tiered-large-policy.py`. Preserve the checkpoint, dataset, cost, and policy hashes together.
5. Deploy the patched Switchyard runtime with the matching checkpoint and policy, then evaluate the reserved test once using `tiered-heldout-live.py`. Check each script's arguments and paths before launching.

See `patches/README.md` for the exact local integration base and patch. The recorded setup is not a claim that an unmodified upstream binary accepts these checkpoints. Live deployment also needs the model-router Python environment and restored model/cache files; node-local storage may be lost when worker nodes are removed.

Generated artifacts live under ignored `data/`, `cache/`, and `checkpoints/`. They must be regenerated or shared separately for exact reproduction; cloning this repository alone does not supply a trained checkpoint. The original final report is `data/shopping-mmlu/tiered-real-large-router-v1/heldout-live-token-v1-report.json`.

The dashboard is local-only and intentionally excluded from the shared repository.

## Offline checks

```bash
python -m unittest discover -s tests -v
```

These checks do not start GPU nodes or make model requests.
