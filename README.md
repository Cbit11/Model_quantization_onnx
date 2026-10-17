# Exporting a Hybrid Conv–Attention–Mamba Vision Model to ONNX Runtime

A small hybrid vision classifier for CIFAR-10 (convolutional stage, then windowed self-attention and a selective state-space (Mamba-style) block), with the selective scan **reimplemented in pure PyTorch** so the whole model can be exported to ONNX and run in ONNX Runtime (ORT). The project covers the full path: train, export, validate numerically, benchmark on CPU and GPU, and break the runtime down by stage.

**Headline result:** ORT cut batch-1 GPU latency by **1.91x** (7.60 → 3.98 ms) and batch-1 CPU latency by **1.61x**, but its throughput was **16% lower than PyTorch at batch 128 on GPU**. The unfused scan accounts for 74–88% of runtime, so it limits what any runtime can gain. Test accuracy is **88.62%** on CIFAR-10.


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
- Trained on CIFAR-10 for `[N]` epochs (SGD, lr `[value]`, batch 128, random-crop + flip). Test accuracy: **88.62%** (validation loss 0.428). 
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
| Max \|logit difference\|, PyTorch vs ORT (CPU provider, one random batch of 4) | **1.55e-06** |
| Max \|logit difference\| (CUDA provider, 10,000 test images) | **6.38e-03** |
| CIFAR-10 test accuracy, PyTorch / ORT (CUDA) | 88.62% / 88.64% |
| Test loss, PyTorch / ORT (CUDA) | 0.4279 / 0.4279 |
| Prediction agreement on 10,000 test images | **99.98%** (2 of 10,000 differ) |

The logit difference on the GPU is larger than on the CPU (6.38e-03 vs 1.55e-06) because of GPU arithmetic such as TF32 convolutions, not the export: the CPU provider matches to about 1e-6 on the same checkpoint. The two rows differ in scope (random batch vs all 10,000 test images). 

## Benchmarks

Setup: AMD Ryzen 7 4800H (8 cores / 16 threads; benchmark used 16 threads ), NVIDIA GeForce RTX 3050 Laptop GPU (4 GB, driver 617.14), WSL2 Ubuntu, Python 3.12, PyTorch 2.10.0+cu128 (CUDA 12.8, cuDNN 9.10), ONNX Runtime 1.30.0 (`onnxruntime-gpu`), ONNX 1.23.1, fp32. Median latency over 200 runs (batch 1) or 30 runs (batch 128) after 20 warm-up runs. CPU thread counts matched between PyTorch and ORT. GPU timings exclude host–device copies (ORT uses IO binding) and synchronize around each call. Reproduce with `python benchmark.py --ckpt model_XXXX --onnx mamba_vision_cifar10.onnx`.

### End-to-end

| Device | Batch | PyTorch (ms) | ORT (ms) | ORT speedup | PyTorch img/s | ORT img/s |
|---|---|---|---|---|---|---|
| CPU | 1 | 4.50 | 2.79 | **1.61x** | 222 | 358 |
| CPU | 128 | 205.6 | 191.3 | **1.07x** | 623 | 669 |
| GPU | 1 | 7.60 | 3.98 | **1.91x** | 132 | 251 |
| GPU | 128 | 13.11 | 15.56 | **0.84x** | 9,763 | 8,226 |

### Where the time goes (ORT speedup per stage)

| Stage | CPU bs 1 | CPU bs 128 | GPU bs 1 | GPU bs 128 |
|---|---|---|---|---|
| Patch embed | 3.67x | 4.03x | 1.62x | 1.00x |
| Level 0 (conv) | 2.11x | 1.38x | 1.48x | 0.85x |
| Level 1 (attention + Mamba) | 1.59x | 1.26x | 2.04x | 0.86x |

The attention + Mamba stage is 87% of PyTorch CPU time and 83% of PyTorch GPU time at batch 128, and 82% of ORT GPU time at batch 1. At batch 128 it peaks at 140 MB of GPU memory, against 16–34 MB for every other stage, because the discretized `(B, L, D, N)` tensors are materialized.

### What the results show

- **Small batches are launch-bound.** Batch 1 on GPU is slower than batch 1 on CPU for PyTorch (7.60 vs 4.50 ms), and batch 128 costs only 1.7x more than batch 1 for 128x the work. The scan loop launches many tiny kernels, and ORT's graph optimization removes some of that overhead, hence 1.91x.
- **Large batches are throughput-bound, and PyTorch wins on GPU.** When the GPU is saturated, PyTorch's kernels beat ORT's on this model (9,763 vs 8,226 img/s, 16% lower for ORT). The result is stable (ORT std 0.22 ms) and repeated across runs, so it is not noise.
- **The scan dominates.** Conv stages speed up well in ORT but are a small share of runtime, so end-to-end gains are limited by the unfused scan.
- **CPU timings are noisy.** PyTorch batch-1 has p95 27.0 ms vs a 4.5 ms median, while ORT is steadier (p95 6.6 ms). The CPU batch-128 speedup varied between roughly 1.0x and 1.4x across runs, so treat it as a range.
- **Tail latency on GPU at batch 1:** ORT's median is half of PyTorch's, but its p95 is higher (14.5 vs 11.5 ms), so quote p95 alongside the median.

## Limitations

- The scan is the unfused reference form, so absolute GPU throughput is below what the official fused kernel would reach. I did not benchmark against `mamba-ssm` 
- The model is small and CIFAR-10-scale; sequence length per scan is 16, so the scan's long-sequence advantages are not exercised.
- fp32 only so far. 

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
