# Switchyard + Dynamo shopping router

This repository is a serving reference for the Amazon Ads shopping demo. Switchyard's trained prefill router chooses between Gemma 4 E4B (`efficient`) and Gemma 4 31B (`capable`) for each Shopping MMLU question. Both models have separate Dynamo Graph Deployments (DGDs) on one two-GPU `g7e.12xlarge`. Each GPU runs one model worker and one colocated copy of the Qwen3.5-0.8B prefill encoder/router; a Kubernetes Service exposes the two router replicas.

| GPU allocation | Dynamo Graph Deployment | Colocated processes |
| --- | --- | --- |
| One GPU | `efficient` | Gemma 4 E4B worker + Switchyard prefill router |
| Other GPU | `capable` | Gemma 4 31B worker + Switchyard prefill router |

The manifests allocate one GPU to each worker but do not pin numeric GPU IDs. The two router replicas share the same frozen checkpoint, targets, tolerance, and request settings. The Service distributes connections across healthy replicas. It does not guarantee an exact 50/50 request split for every client.

## Reproduce the serving setup

Start with the [deployment guide](docs/shopping-demo.md). It covers model access, pinned model snapshots, building the patched Switchyard runtime, staging the trained checkpoint, applying both DGDs and the Service, and checking both router replicas. The default manifests disable vLLM prefix caching. An **opt-in, version-guarded experimental shared-prefix-only cache policy** is included for reproducing the later controlled cache study; it is not a general production recommendation.

The checkpoint, model weights, credentials, source checkouts, and Kubernetes cluster are **not** committed. The repo therefore provides serving code and a deployment recipe, not a turnkey cluster. Training/tuning helpers under `scripts/shopping-mmlu/` document how the routing policy was produced. No AIPerf runner, load-generation dataset, or capacity-benchmark artifacts are included.

## Scope of the measured result

The controlled shared-prefix-only comparison found 93.4% held-out accuracy for Switchyard versus 95.2% for 31B-only. On the same complete G7e instance, selected SLO-compliant throughput was 191.56 versus 122.23 requests/s, implying **36.2% lower measured G7e cost per compliant request**. This used 500 held-out questions sampled repeatedly, two-token answers, an experimental 32-token cache cap, and a 1-second latency target met by at least 95% of requests. It is a scoped infrastructure-efficiency measurement, **not** a claim about production Amazon Ads traffic or total deployment cost. The serving manifests in this repo default to cache disabled, so those numbers do not describe their default cache mode.

Upstream: [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard), [Dynamo](https://github.com/ai-dynamo/dynamo), [Shopping MMLU](https://github.com/KL4805/ShoppingMMLU).
