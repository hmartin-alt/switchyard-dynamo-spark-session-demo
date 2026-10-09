# Deploy the Switchyard shopping router

This is the **two-encoder serving topology**, not the earlier single-encoder isolation test. One E4B worker and one 31B worker run in separate Dynamo Graph Deployments (DGDs) on the two GPUs of the same `g7e.12xlarge`; each worker starts a colocated Switchyard prefill router. Both routers target the same Dynamo frontends, use the same checkpoint and tolerance, and are exposed by one Kubernetes Service.

## What you need

- An EKS cluster with the Dynamo operator and a `dynamo` namespace. Label **one prepared two-GPU G7e node** `workload=model-serving` for this example. If more nodes carry that label, add explicit node affinity so both workers and node-local files land on the intended node.
- Access to the gated Gemma model repositories through `hf-token-secret` in `dynamo` with key `HUGGING_FACE_HUB_TOKEN`; do not commit credentials. Verify the Dynamo runtime image `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0` is available.
- A Switchyard checkout at commit `a601a9a3f9db149a1ad430fa43b1463a170c8a82` and the compatible model-router toolkit source required by [the runtime patches](../patches/README.md). These source trees are not vendored here.
- The trained shopping router checkpoint, staged as `/opt/switchyard-colocated/prefill_router.pt` on the serving node. It is not included in this repo. The demo checkpoint's SHA-256 is `5c3e0704368f248a469d6762252bb3a1257bfd7016c885417de1707488247bdf`; verify it before use. Use the checkpoint and tolerance from the same frozen policy; do not substitute an unverified checkpoint.
- `kubectl` access and enough local storage for both pinned model snapshots and the built runtime. The `hostPath` files are node-local and do not survive node replacement automatically.

## 1. Prepare models and runtime

Apply the [model prefetch Job](../manifests/dynamo/prefetch-job.yaml) on the prepared node. It downloads the pinned E4B and 31B Hugging Face revisions used by the launchers. Wait for successful completion and verify both expected snapshot directories exist under `/opt/hf-cache/hub` on that node. The Job has a short TTL, so inspect errors promptly.

```bash
kubectl apply -f manifests/dynamo/prefetch-job.yaml
kubectl -n dynamo wait --for=condition=complete job/prefetch-models --timeout=60m
kubectl apply -f manifests/router/colocated-builder.yaml
kubectl -n dynamo wait --for=condition=Ready pod/switchyard-colocated-builder --timeout=10m
```

Apply the [builder Pod](../manifests/router/colocated-builder.yaml) on that same node. Copy the two prepared source trees into its `/runtime/switchyard-src` and `/runtime/llm-router-src` directories, this repo's `patches/` directory to `/runtime/patches`, and [the restore script](../scripts/shopping-mmlu/restore-colocated-runtime.sh) to `/runtime/restore-colocated-runtime.sh`. Run the restore script **once against fresh source**. It checks and applies the Switchyard integration patch and toolkit patches, installs runtime Python dependencies, and builds `/runtime/switchyard-server-profiled`. See [patch instructions](../patches/README.md) for the exact order.

From this repo's root, after preparing the two source directories and checkpoint locally:

```bash
kubectl -n dynamo cp /path/to/switchyard-src switchyard-colocated-builder:/runtime/switchyard-src
kubectl -n dynamo cp /path/to/llm-router-src switchyard-colocated-builder:/runtime/llm-router-src
kubectl -n dynamo cp patches switchyard-colocated-builder:/runtime/patches
kubectl -n dynamo cp scripts/shopping-mmlu/restore-colocated-runtime.sh switchyard-colocated-builder:/runtime/restore-colocated-runtime.sh
kubectl -n dynamo exec switchyard-colocated-builder -- bash /runtime/restore-colocated-runtime.sh
kubectl -n dynamo cp manifests/router/colocated-routes.toml switchyard-colocated-builder:/runtime/routes.toml
kubectl -n dynamo cp manifests/router/start-efficient-router.sh switchyard-colocated-builder:/runtime/start-efficient-router.sh
kubectl -n dynamo cp manifests/router/start-capable-router.sh switchyard-colocated-builder:/runtime/start-capable-router.sh
kubectl -n dynamo cp /path/to/prefill_router.pt switchyard-colocated-builder:/runtime/prefill_router.pt
kubectl -n dynamo exec switchyard-colocated-builder -- sha256sum /runtime/prefill_router.pt
kubectl -n dynamo exec switchyard-colocated-builder -- chmod +x /runtime/start-efficient-router.sh /runtime/start-capable-router.sh
```

Stage these files on the same node (the builder's `/runtime` is the workers' `/router-runtime`):

| Repo file or supplied artifact | Node-local destination |
| --- | --- |
| [routes](../manifests/router/colocated-routes.toml) | `/opt/switchyard-colocated/routes.toml` |
| [E4B launcher](../manifests/router/start-efficient-router.sh) | `/opt/switchyard-colocated/start-efficient-router.sh` |
| [31B launcher](../manifests/router/start-capable-router.sh) | `/opt/switchyard-colocated/start-capable-router.sh` |
| Your frozen router checkpoint | `/opt/switchyard-colocated/prefill_router.pt` |

Both launchers must be executable. Before deployment, verify the checkpoint is the one selected by your validation run, the routes' `tolerance = 0.1760250329971314` is intentional, and the model snapshot paths exist. The launchers deliberately fail rather than silently changing cache mode.

## 2. Deploy and check

```bash
kubectl apply -f manifests/dynamo/dgd-efficient-colocated.yaml
kubectl apply -f manifests/dynamo/dgd-capable.yaml
kubectl apply -f manifests/router/colocated-service.yaml
kubectl -n dynamo get dgd efficient capable
kubectl -n dynamo get pods -l app=switchyard-production-rr -o wide
kubectl -n dynamo get endpointslices -l kubernetes.io/service-name=switchyard-production-rr
```

Expect **two ready worker Pods**, one labeled `switchyard-replica=efficient` and one `switchyard-replica=capable`, and two ready Service endpoints on the same prepared node. The worker readiness probe requires both local Dynamo/vLLM and the local router to respond. Inspect each worker's logs if it is not ready. You can port-forward `svc/switchyard-production-rr` on port 4000 and send a chat completion to `/v1/chat/completions` using the served route name `shopping`; use the fixed request settings required by your application. Check each replica's routing log separately if you need to confirm traffic reaches both.

The Service distributes TCP connections, not guaranteed individual requests. Client connection reuse can skew the split. Use client-side connection rotation or a request-aware proxy if balanced **per-request** distribution is required; do not infer a 50/50 split from the Service name.

## Cache modes

The manifests set `SWITCHYARD_PREFIX_CACHE_MODE=disabled` on **both** workers, which passes `--no-enable-prefix-caching` to each `dynamo.vllm` process. This was the conservative full-prefill comparison mode.

For a controlled reproduction of the later shared-prefix-only study, first stage [the experimental vLLM patch](../manifests/router/experimental/prefix-only-vllm-patch.py) as `/opt/switchyard-colocated/experimental/prefix-only-vllm-patch.py`:

```bash
kubectl -n dynamo exec switchyard-colocated-builder -- mkdir -p /runtime/experimental
kubectl -n dynamo cp manifests/router/experimental/prefix-only-vllm-patch.py switchyard-colocated-builder:/runtime/experimental/prefix-only-vllm-patch.py
```

Then change `SWITCHYARD_PREFIX_CACHE_MODE` to `shared-prefix-only` in **both** DGD manifests before deploying. The patch allows reuse of at most the first 32 prompt tokens and refuses to run if the installed vLLM source does not match its expected version. This was an experimental benchmark policy to avoid complete-cache hits from repeated questions; it is **not** ordinary production prefix caching. Test and review it separately before considering it for another deployment.

## Policy and dataset provenance

The shopping policy was tuned on 3,500 training questions and 500 validation questions, then evaluated on 500 held-out Shopping MMLU questions. The held-out split is intentionally easy-heavy (400 easy, 100 hard), not representative of all Amazon Ads traffic. The router estimates each model's probability of a correct answer and selects between them using the frozen tolerance. `configs/prefill-router/` and `scripts/shopping-mmlu/` contain optional training, policy-selection, and accuracy-evaluation helpers; they are **not** load-testing tools.

The token-price inputs in `configs/prefill-router/shopping-costs-token-proxy.json` served the original **routing policy selection**. They are not the infrastructure-cost calculation. The measured G7e cost comparison uses the same complete instance for each configuration and divides its hourly rate by measured SLO-compliant goodput. It excludes the CPU load-generator, EKS control plane, storage, network, and operational overhead. Do not present it as an AWS bill or production saving.

Run offline checks without a cluster:

```bash
python3 -m unittest discover -s tests -v
```
