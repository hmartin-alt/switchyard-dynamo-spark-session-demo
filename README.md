# Shopping routing with Switchyard + Dynamo

Switchyard chooses between Gemma 4 E4B (efficient) and Gemma 4 31B (capable) for shopping questions. A Qwen3.5-0.8B prefill encoder and trained routing head predict each model's likelihood of answering correctly. The policy balances that prediction against estimated token cost.

Both models run on Dynamo. The encoder shares the efficient model's GPU.

## Results

On 500 held-out Shopping MMLU questions:

| Metric | E4B only | 31B only | Switchyard |
| --- | ---: | ---: | ---: |
| Accuracy | 83.2% | 95.2% | 93.6% |
| Mean request latency | 0.197s | 0.404s | 0.350s |
| Estimated token cost / 1,000 requests | $0.00240 | $0.01095 | $0.00440 |

<<<<<<< HEAD
77.2% of requests went to E4B. Estimated token-cost savings were 59.9% versus 31B-only, with 1.6 percentage points lower accuracy.

Data: 3,500 training, 500 validation, 500 held-out questions.
=======
Switchyard sent 77.2% of requests to E4B, with 1.6 percentage points lower accuracy than 31B-only. Under the saved pricing assumptions, estimated token cost was 59.9% lower.

Cost figures exclude router compute and infrastructure; they are not EC2 bill savings. The E4B price is corroborated by a third-party listing but has not been reverified directly with DeepInfra. See the [pricing assumptions](docs/shopping-demo.md#cost-policy).
>>>>>>> 61dc6b3 (Clean up docs to only include necessary relevant information)

## Getting started

Follow the [setup and workflow guide](docs/shopping-demo.md).

- `scripts/shopping-mmlu/` — dataset preparation, evaluation, and policy selection.
- `configs/prefill-router/` — training configuration and pricing assumptions.
- `manifests/` — GPU training and Dynamo/Switchyard deployment.
- `patches/` — required local runtime changes.
- `tests/` — offline checks.

<<<<<<< HEAD
=======
The dataset is an intentionally easy-heavy shopping workload, not a representative sample of all Amazon Ads traffic. Data, checkpoints, model weights, credentials, and the dashboard are not included.

>>>>>>> 61dc6b3 (Clean up docs to only include necessary relevant information)
Upstream: [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard), [Dynamo](https://github.com/ai-dynamo/dynamo), [Shopping MMLU](https://github.com/KL4805/ShoppingMMLU).
