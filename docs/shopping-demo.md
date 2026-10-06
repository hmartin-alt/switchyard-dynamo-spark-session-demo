# Setup and workflow

## Data

We use real Shopping MMLU questions with their original choices and answers:

| Split | Easy | Hard | Total |
| --- | ---: | ---: | ---: |
| Training | 2,850 | 650 | 3,500 |
| Validation | 400 | 100 | 500 |
| Held-out test | 400 | 100 | 500 |

Easy questions cover attributes, categories, compatibility, and commonsense. Hard questions require product arithmetic or unit conversion. Previously inspected pilot questions are restricted to training.

Validation selects the routing policy; the held-out test measures the frozen setup. See the [README](../README.md#results) for results. A random router using the same model mix has expected accuracy of 85.9%; this is a calculated baseline, not a live randomized run.

## Cost policy

The policy balances predicted correctness against estimated model cost:

`score = lambda × predicted correctness − (1 − lambda) × relative cost`

Each model's typical request cost comes from its median training input/output token counts multiplied by the saved token prices. Dividing both costs by the capable model's cost makes capable cost 1 and efficient cost a smaller fraction.

Validation selected lambda approximately 0.816, equivalent to Switchyard tolerance approximately 0.176. This lets Switchyard choose E4B when its predicted success is close enough to 31B's. Exact serving values live in `manifests/router/colocated-routes.toml` and `start-colocated.sh`; tolerance is not a guarantee about actual accuracy loss.

Final cost reporting uses actual response token counts, not training medians.

The saved pricing assumptions in `configs/prefill-router/shopping-costs-token-proxy.json` are:

| Model | Input / million tokens | Output / million tokens |
| --- | ---: | ---: |
| E4B | $0.02 | $0.10 |
| 31B | $0.09 | $0.34 |

The E4B rate is supported by a [third-party DeepInfra listing](https://computeprices.com/providers/deep-infra/models/gemma-4-e4b), but could not be reverified directly on DeepInfra. The 31B rate is the experiment's saved OpenRouter assumption, not a current quote. Savings are conditional on these rates and exclude encoder compute, idle GPUs, storage, and networking.

## Requirements

- Python 3.10+, the model-router toolkit with prefill/training dependencies, OpenAI's Python client, and Matplotlib.
- EKS with the Dynamo operator and sufficient GPU capacity.
- Access to both Gemma models, configured through `hf-token-secret` in namespace `dynamo`.
- The [local Switchyard runtime changes](../patches/README.md).
- Dataset and checkpoint artifacts, generated locally or supplied separately.

## Workflow

1. **Build the dataset:** run `tiered-large.py --archive <data.zip> --output <new-directory>` from `scripts/shopping-mmlu/`. The builder requires the audited official archive and the original prior-pilot inputs for its exclusion list.
2. **Collect training labels:** run `tiered-large-evaluate.py`, then `tiered-large-router.py`.
3. **Train:** use the toolkit's training command with `configs/prefill-router/shopping-pool.yaml` and the training CSV. Save the checkpoint under `checkpoints/shopping-tiered-real-large-v1/`.
4. **Select the policy:** record the checkpoint's SHA256 in `checkpoint.json` under the router artifact directory, then run `tiered-token-cost-sweep.py` on validation and freeze the result with `tiered-large-policy.py`.
5. **Deploy:** restore the runtime and model caches, then deploy the models and router using the files below. Match the serving checkpoint and tolerance to the frozen policy.
6. **Evaluate:** run `tiered-heldout-live.py` against the direct-model and Switchyard endpoints with fresh output paths.

Scripts contain experiment-specific paths. Check their arguments and defaults before running; do not overwrite saved evidence. The checkpoint and policy metadata are used by the scripts to reject mismatched artifacts.

## Deployment files

| File | Purpose |
| --- | --- |
| `manifests/dynamo/prefetch-job.yaml` | Download model weights. |
| `manifests/router/train-large.yaml` | GPU training pod. |
| `manifests/router/colocated-builder.yaml` | Runtime-building pod. |
| `manifests/dynamo/dgd-efficient-colocated.yaml` | E4B and colocated encoder/router. |
| `manifests/dynamo/dgd-capable.yaml` | 31B workers; currently requests two replicas. |
| `manifests/router/colocated-service.yaml` | Expose the router endpoint. |
| `manifests/router/colocated-routes.toml` | Target endpoints and routing policy. |
| `manifests/router/start-colocated.sh` | Start the efficient model and router. |

Apply resources in namespace `dynamo`. These manifests use node-local storage: prepare the model snapshots and runtime on every eligible node, or pin pods to a prepared node. Check GPU capacity and snapshot paths before deployment. Node removal can erase these files.

## Checks

```bash
python -m unittest discover -s tests -v
```

These are offline checks; they do not start GPUs or send inference requests.
