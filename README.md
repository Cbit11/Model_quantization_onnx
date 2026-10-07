# Exporting a Hybrid Conv–Attention–Mamba Vision Model to ONNX Runtime

A small hybrid vision classifier for CIFAR-10 (convolutional stage, then windowed self-attention and a selective state-space (Mamba-style) block), with the selective scan **reimplemented in pure PyTorch** so the whole model can be exported to ONNX and run in ONNX Runtime (ORT). The project covers the full path: train, export, validate numerically, benchmark on CPU and GPU, and break the runtime down by stage.

**Headline result:** ORT cut batch-1 GPU latency by **1.84x** (6.97 → 3.79 ms) and batch-1 CPU latency by **1.39x**, but was **16% slower than PyTorch at batch 128 on GPU**. The unfused scan accounts for 65–87% of runtime, so it limits what any runtime can gain.



---

## Why this is non-trivial

The official `mamba-ssm` package computes the selective scan with a fused CUDA/Triton kernel. `torch.onnx.export` works by tracing standard PyTorch ops, so it cannot see inside a custom compiled kernel, and the official block does not export.

This repo rewrites the scan as ordinary PyTorch ops (a loop over the sequence), which traces to ONNX. The cost is that the scan is no longer fused: each step launches separate kernels and the discretized tensors are materialized in memory. The benchmarks quantify what that costs.

## Model

| Stage | What it is |
|---|---|
| Patch embed | Two stride-2 conv + BatchNorm + ReLU layers (32×32 → 8×8, 96 channels) |
| Level 0 | Conv residual block, then stride-2 downsample (8×8 → 4×4, 192 channels) |
| Level 1 | 2 hybrid blocks on 4×4 windows: windowed multi-head attention + selective SSM (Mamba-style) + MLPs |
| Head | BatchNorm → global average pool → linear (10 classes) |

- Selective SSM follows Algorithm 2 (S6) of the Mamba paper: input-dependent Δ, B, C; learned A and D; zero-order-hold discretization; sequential scan over the flattened window (`d_state = 16`).
- Config: `dims=96, depths=[1,2], window_size=[8,4], num_heads=[6,6], mlp_ratio=0.6`. `[N]` parameters.
- Trained on CIFAR-10 for `80` epochs (SGD, lr `1e-4`, batch 128, random-crop + flip). Test accuracy: **`[83.5]%`**.


## Export

```bash
python export_onnx.py --ckpt model_XXXX --out mamba_vision_cifar10.onnx
```

Opset 18, dynamic batch axis. Problems solved along the way:

- **Dropout in `eval()`:** `F.scaled_dot_product_attention` applies the `dropout_p` it is given regardless of `model.eval()`, so PyTorch kept dropping attention weights at inference while the ONNX graph did not. Fix: `dropout_p = p if self.training else 0.0`.
- **Batch size baked into the graph:** `int(windows.shape[0] / ...)` in `window_reverse` made the exporter treat the batch as a constant. Fix: let `view` infer it with `-1`.
- **Opset:** the dynamo exporter emits opset 18; requesting 17 triggers a downconversion that fails on some reduce ops. Export at 18 directly.

## Validation

| Check | Result |
|---|---|
| Max \|logit difference\|, PyTorch vs ORT (CPU, batch 4) | **3.81e-06** |
| Max \|logit difference\| (CUDA provider) | `[run validate.py]` |
| CIFAR-10 test accuracy, PyTorch / ORT | `[XX.XX%]` / `[XX.XX%]` |
| Prediction agreement on 10,000 test images | `[XX.XX%]` |

## Benchmarks

Setup: `[CPU model, cores/threads used]`, `[GPU, e.g. RTX 3050 Laptop/Desktop, VRAM]`, WSL2 Ubuntu, PyTorch `[ver]`, ONNX Runtime `[ver]` (`onnxruntime-gpu`), fp32. Median latency over 200 runs (batch 1) or 30 runs (batch 128) after 20 warm-up runs. CPU thread counts matched between PyTorch and ORT. GPU timings exclude host–device copies (ORT uses IO binding) and synchronize around each call. Reproduce with `python benchmark.py --ckpt model_XXXX --onnx mamba_vision_cifar10.onnx`.

### End-to-end

| Device | Batch | PyTorch (ms) | ORT (ms) | ORT speedup | PyTorch img/s | ORT img/s |
|---|---|---|---|---|---|---|
| CPU | 1 | 4.70 | 3.39 | **1.39x** | 213 | 295 |
| CPU | 128 | 239.0 | 170.4 | **1.40x** | 536 | 751 |
| GPU | 1 | 6.97 | 3.79 | **1.84x** | 144 | 264 |
| GPU | 128 | 12.42 | 14.74 | **0.84x** | 10,303 | 8,685 |

### Where the time goes (ORT speedup per stage)

| Stage | CPU bs 1 | CPU bs 128 | GPU bs 1 | GPU bs 128 |
|---|---|---|---|---|
| Patch embed | 2.74x | 2.98x | 2.13x | 1.16x |
| Level 0 (conv) | 1.86x | 0.97x | 1.66x | 0.84x |
| Level 1 (attention + Mamba) | 1.50x | 1.18x | 2.00x | 0.85x |

The attention + Mamba stage is 77% of PyTorch CPU time and 87% of PyTorch GPU time at batch 128, and 80% of ORT GPU time at batch 1. At batch 128 it peaks at 140 MB of GPU memory, against 16–34 MB for every other stage, because the discretized `(B, L, D, N)` tensors are materialized.

### What the results show

- **Small batches are launch-bound.** Batch 1 on GPU is slower than batch 1 on CPU for PyTorch (6.97 vs 4.70 ms), and batch 128 costs only 1.8x more than batch 1 for 128x the work. The scan loop launches many tiny kernels, and ORT's graph optimization removes some of that overhead, hence 1.84x.
- **Large batches are throughput-bound, and PyTorch wins on GPU.** When the GPU is saturated, PyTorch's kernels beat ORT's on this model by about 16%. The result is stable (ORT std 0.12 ms), not noise.
- **The scan dominates.** Conv stages speed up well in ORT but are a small share of runtime, so end-to-end gains are limited by the unfused scan.
- **CPU batch-1 PyTorch timings are noisy** (p95 20 ms vs 4.7 ms median) while ORT is steady (p95 5.0 ms). Treat CPU numbers as ±`[X]%` across repeated runs.

## Limitations

- The scan is the unfused reference form, so absolute GPU throughput is below what the official fused kernel would reach. I did not benchmark against `mamba-ssm` `[or: see results in ...]`.
- `[Known model issues: list any of these that apply to the checkpoint, e.g. the A-matrix sign/ordering in discretization, missing MLP residual connections. Say whether you fixed them and retrained.]`
- The model is small and CIFAR-10-scale; sequence length per scan is 16, so the scan's long-sequence advantages are not exercised.
- fp32 only so far. `[INT8 quantization results go here if completed.]`

## Future work

- CUDA graphs (ORT `enable_cuda_graph` / `torch.compile(mode="reduce-overhead")`) to cut batch-1 launch overhead.
- Rewrite the scan to compute the discretization inside the loop and avoid materializing `(B, L, D, N)` tensors.
- ORT cuDNN conv algorithm search for the conv stages.
- Dynamic INT8 quantization and an accuracy/latency trade-off table.
- Compare against the fused `mamba-ssm` kernel.

## Repository layout

```
model.py            hybrid model + pure-PyTorch selective scan
train.py            CIFAR-10 training
export_onnx.py      ONNX export (opset 18, dynamic batch)
validate.py         numerical + accuracy parity, PyTorch vs ORT   [add]
benchmark.py        latency / throughput / per-stage benchmark
bench_results/      results.csv, results.md
```

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023).
- Hatamizadeh & Kautz, *MambaVision: A Hybrid Mamba-Transformer Vision Backbone* (2024).
