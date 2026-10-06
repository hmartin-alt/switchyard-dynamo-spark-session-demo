# Shopping MMLU routing with Switchyard + Dynamo

A trained Qwen3.5-0.8B prefill encoder predicts whether Gemma 4 E4B or Gemma 4 31B can answer a shopping question. Switchyard selects a model using a validation-tuned cost–quality policy. Both models run on Dynamo; the encoder shares the efficient model's GPU.

## Held-out results

| Metric | E4B only | 31B only | Switchyard |
| --- | ---: | ---: | ---: |
| Accuracy | 83.2% | 95.2% | 93.6% |
| Mean latency | 0.197s | 0.404s | 0.350s |
| Estimated token cost / 1,000 requests | $0.00240 | $0.01095 | $0.00440 |

77.2% of requests went to E4B. Estimated token-cost savings were 59.9% versus 31B-only, with 1.6 percentage points lower accuracy.

Data: 3,500 training, 500 validation, 500 held-out questions.

## Start here

Read [the setup and experiment guide](docs/shopping-demo.md).

- `scripts/shopping-mmlu/`: build data, collect labels, sweep policy, and evaluate.
- `configs/prefill-router/`: training configuration and token-price snapshot.
- `manifests/`: Dynamo deployments, GPU training, and colocated router runtime.
- `patches/`: required local Switchyard/toolkit integration and batching changes.
- `tests/`: focused offline checks.

Upstream: [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard), [Dynamo](https://github.com/ai-dynamo/dynamo), [Shopping MMLU](https://github.com/KL4805/ShoppingMMLU).
