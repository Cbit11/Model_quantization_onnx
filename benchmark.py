#!/usr/bin/env python
"""
Benchmark PyTorch vs ONNX Runtime for the hybrid conv / attention / Mamba CIFAR-10 model.

What it measures
  1. Parity check (PyTorch vs ORT logits) before any timing.
  2. End-to-end latency + throughput: {PyTorch, ORT} x {CPU, GPU} x batch sizes.
  3. Per-stage breakdown (patch_embed, each level, head): each stage is exported to its
     own static-batch ONNX file, so you can see where ORT helps and where the unrolled
     Mamba scan dominates.

Usage
  python benchmark.py --ckpt model_XXXX --onnx mamba_vision_cifar10.onnx
  python benchmark.py --ckpt checkpoint_9.pth --batch-sizes 1 32 128 --threads 8
  python benchmark.py --ckpt model_XXXX --no-stages --skip-gpu

Outputs (in --out-dir): results.csv, results.md (paste the table into your README).

Fairness notes
  * GPU timings exclude host<->device copies: inputs are already on the GPU for both
    backends (ORT uses IO binding on a torch CUDA tensor).
  * CPU thread counts are matched (torch.set_num_threads == ORT intra_op_num_threads).
  * PyTorch runs under torch.inference_mode(); cuda.synchronize() brackets every GPU call.
"""
import argparse
import csv
import os
import platform
import time
import warnings
import tempfile

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from model import final_model

warnings.filterwarnings("ignore")

# Must match the config used for training.
CFG = dict(
    dims=96, depths=[1, 2], mlp_ratio=0.6, window_size=[8, 4], num_classes=10,
    drop_rate=0.6, drop_path_rate=0.4, attn_drop_rate=0.6, num_heads=[6, 6],
)
IMG_SHAPE = (3, 32, 32)


# --------------------------------------------------------------------------- helpers
def load_model(ckpt_path):
    model = final_model(**CFG)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.eval()


class Head(nn.Module):
    """norm -> avgpool -> flatten -> linear, shares weights with the full model."""

    def __init__(self, m):
        super().__init__()
        self.norm, self.pool, self.head = m.norm, m.avgpool, m.head

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.norm(x)), 1))


def time_fn(fn, warmup, iters, sync=None):
    """Return per-call latencies in ms."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(times)


def summarize(times_ms, bs):
    med = float(np.median(times_ms))
    return dict(
        median_ms=med,
        mean_ms=float(times_ms.mean()),
        p95_ms=float(np.percentile(times_ms, 95)),
        std_ms=float(times_ms.std()),
        img_per_s=bs / (med / 1000.0),
    )


def make_session(path, device, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if device == "cpu":
        so.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
    else:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return None
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(path, so, providers=providers)
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        return None
    return sess


def ort_gpu_fn(sess, x_gpu):
    """Run ORT on a CUDA torch tensor with IO binding (no host copies)."""
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    binding = sess.io_binding()
    binding.bind_input(
        name=in_name, device_type="cuda", device_id=0, element_type=np.float32,
        shape=tuple(x_gpu.shape), buffer_ptr=x_gpu.data_ptr(),
    )
    binding.bind_output(out_name, "cuda", 0)

    def run():
        sess.run_with_iobinding(binding)
        if hasattr(binding, "synchronize_outputs"):
            binding.synchronize_outputs()

    return run


def bench_torch(mod, x_cpu, device, warmup, iters):
    mod = mod.to(device).eval()
    x = x_cpu.to(device)
    sync = torch.cuda.synchronize if device == "cuda" else None
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        t = time_fn(lambda: mod(x), warmup, iters, sync)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    return t, mem


def bench_ort(sess, x_cpu, device, warmup, iters):
    if device == "cuda":
        x = x_cpu.to("cuda").contiguous()
        fn = ort_gpu_fn(sess, x)
    else:
        name = sess.get_inputs()[0].name
        arr = np.ascontiguousarray(x_cpu.numpy())
        fn = lambda: sess.run(None, {name: arr})
    return time_fn(fn, warmup, iters, None)


def export_static(mod, x, path):
    """Export one stage with a fixed batch size. Legacy exporter first (most robust)."""
    mod = mod.cpu().eval()
    try:
        torch.onnx.export(mod, (x,), path, opset_version=17, dynamo=False,
                          input_names=["input"], output_names=["output"])
    except Exception:
        torch.onnx.export(mod, (x,), path, opset_version=18, dynamo=True,
                          input_names=["input"], output_names=["output"])


def build_stages(model, bs):
    x = torch.randn(bs, *IMG_SHAPE)
    stages = []
    with torch.no_grad():
        stages.append(("patch_embed", model.patch_embed, x))
        h = model.patch_embed(x)
        for i, level in enumerate(model.levels):
            kind = "conv" if level.conv else "attn+mamba"
            stages.append((f"level{i}_{kind}", level, h.clone()))
            h = level(h)
        stages.append(("head", Head(model), h.clone()))
    return stages


# --------------------------------------------------------------------------- reporting
def write_outputs(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "device", "batch", "backend", "median_ms", "mean_ms",
                    "p95_ms", "std_ms", "img_per_s", "peak_gpu_mem_mb"])
        for (scope, device, bs), d in results.items():
            for backend, (st, mem) in d.items():
                w.writerow([scope, device, bs, backend, f"{st['median_ms']:.4f}",
                            f"{st['mean_ms']:.4f}", f"{st['p95_ms']:.4f}",
                            f"{st['std_ms']:.4f}", f"{st['img_per_s']:.1f}",
                            "" if mem is None else f"{mem:.1f}"])

    hdr = ("| Scope | Device | Batch | PyTorch median (ms) | ORT median (ms) | Speedup | "
           "PyTorch img/s | ORT img/s | PyTorch p95 (ms) | ORT p95 (ms) |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for (scope, device, bs), d in results.items():
        pt = d.get("PyTorch", (None, None))[0]
        ot = d.get("ORT", (None, None))[0]
        f = lambda s, k, p=3: "n/a" if s is None else f"{s[k]:.{p}f}"
        speed = "n/a" if (pt is None or ot is None) else f"{pt['median_ms'] / ot['median_ms']:.2f}x"
        lines.append(
            f"| {scope} | {device.upper()} | {bs} | {f(pt, 'median_ms')} | {f(ot, 'median_ms')} | "
            f"{speed} | {f(pt, 'img_per_s', 1)} | {f(ot, 'img_per_s', 1)} | "
            f"{f(pt, 'p95_ms')} | {f(ot, 'p95_ms')} |"
        )
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nSaved: {out_dir}/results.csv, {out_dir}/results.md")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="state_dict or training checkpoint (.pth)")
    ap.add_argument("--onnx", default="mamba_vision_cifar10.onnx")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 128])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters-small", type=int, default=200, help="iterations when batch == 1")
    ap.add_argument("--iters-large", type=int, default=30, help="iterations when batch > 1")
    ap.add_argument("--no-stages", action="store_true", help="skip per-stage breakdown")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--out-dir", default="bench_results")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    use_cuda = torch.cuda.is_available() and not args.skip_gpu
    print(f"torch {torch.__version__} | onnxruntime {ort.__version__} | python {platform.python_version()}")
    print(f"CPU threads: {args.threads} | ORT providers: {ort.get_available_providers()}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(args.ckpt)

    # ---- parity check (also confirms the batch axis is dynamic: uses batch 4)
    sess_cpu = make_session(args.onnx, "cpu", args.threads)
    in_name = sess_cpu.get_inputs()[0].name
    xp = torch.randn(4, *IMG_SHAPE)
    with torch.inference_mode():
        ref = model(xp).numpy()
    out = sess_cpu.run(None, {in_name: xp.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    print(f"[parity] max |logit diff| PyTorch vs ORT (batch 4): {diff:.2e}")
    if diff > 1e-3:
        print("WARNING: parity is worse than 1e-3. Timings below compare models that "
              "do not match; fix the export before trusting them.")

    sess_gpu = make_session(args.onnx, "cuda", args.threads) if use_cuda else None
    if use_cuda and sess_gpu is None:
        print("NOTE: CUDAExecutionProvider unavailable (install onnxruntime-gpu). "
              "Skipping ORT-GPU rows; PyTorch-GPU still runs.")

    # ---- per-stage static exports (done on CPU before any module moves to the GPU)
    stage_data = {}
    tmp = tempfile.mkdtemp(prefix="stage_onnx_")
    if not args.no_stages:
        for bs in args.batch_sizes:
            entries = []
            for name, mod, x in build_stages(model, bs):
                path = os.path.join(tmp, f"{name.replace('+', '_')}_bs{bs}.onnx")
                try:
                    export_static(mod, x, path)
                except Exception as e:
                    print(f"[stage export] {name} bs={bs} failed: {type(e).__name__}: {str(e)[:120]}")
                    path = None
                entries.append((name, mod, x, path))
            stage_data[bs] = entries

    devices = ["cpu"] + (["cuda"] if use_cuda else [])
    results = {}

    def record(scope, device, bs, backend, times, mem=None):
        results.setdefault((scope, device, bs), {})[backend] = (summarize(times, bs), mem)

    for device in devices:
        sess = sess_cpu if device == "cpu" else sess_gpu
        for bs in args.batch_sizes:
            iters = args.iters_small if bs == 1 else args.iters_large
            print(f"\n=== {device.upper()} | batch {bs} | end-to-end ===")
            x = torch.randn(bs, *IMG_SHAPE)
            t, mem = bench_torch(model, x, device, args.warmup, iters)
            record("end-to-end", device, bs, "PyTorch", t, mem)
            print(f"PyTorch  median {np.median(t):8.3f} ms")
            if sess is not None:
                t = bench_ort(sess, x, device, args.warmup, iters)
                record("end-to-end", device, bs, "ORT", t)
                print(f"ORT      median {np.median(t):8.3f} ms")

            for name, mod, xs, path in stage_data.get(bs, []):
                scope = f"stage: {name}"
                t, mem = bench_torch(mod, xs, device, args.warmup, iters)
                record(scope, device, bs, "PyTorch", t, mem)
                if path is not None:
                    ssess = make_session(path, device, args.threads)
                    if ssess is not None:
                        t = bench_ort(ssess, xs, device, args.warmup, iters)
                        record(scope, device, bs, "ORT", t)

    write_outputs(results, args.out_dir)


if __name__ == "__main__":
    main()