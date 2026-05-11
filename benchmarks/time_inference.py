"""
Benchmark model inference speed.

Usage:
    python benchmarks/time_inference.py --run_dir outputs/2025-01-15/10-30-45
    python benchmarks/time_inference.py --run_dir outputs/2025-01-15/10-30-45 --n_iterations 50 --batch_size 4
    python benchmarks/time_inference.py --run_dir outputs/2025-01-15/10-30-45 --steps 16
"""

import time
import argparse
import numpy as np
from pathlib import Path

import torch

from dataset.echodataset import EchoDataset
from evaluation.functions import load_model_from_run, _get_run_config


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model inference speed")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory (.hydra/config.yaml + checkpoints/)")
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="Checkpoint filename (default: last.ckpt or newest)")
    parser.add_argument("--n_iterations", type=int, default=100,
                        help="Number of timed iterations (default: 100)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations excluded from timing (default: 5)")
    parser.add_argument("--n_missing_frames", default='max',
                        help="Frame masking config (default: 'max')")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of sampling steps (default: 1; use 1 for RMMFlow)")
    return parser.parse_args()


def _create_batch(cfg, batch_size, n_missing_frames, device):
    ds = EchoDataset(cfg, split='test', n_missing_frames=n_missing_frames, cache=False)
    sample = ds[0]
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).repeat(batch_size, *[1] * value.ndim).to(device)
        else:
            batch[key] = value
    return batch


@torch.no_grad()
def _benchmark(model, batch, data_shape, steps, n_iterations, warmup):
    device = next(model.parameters()).device

    print(f"\n{'='*60}")
    print(f"Benchmark: {n_iterations} iterations (+{warmup} warmup)")
    print(f"Batch size: {batch['cond_image'].shape[0]}")
    print(f"Data shape: {data_shape}")
    print(f"Sampling steps: {steps}")
    print(f"{'='*60}\n")

    def _infer():
        return model.sample(
            encoder_hidden_states=batch['encoder_hidden_states'],
            cond_image=batch['cond_image'],
            batch_size=batch['cond_image'].shape[0],
            data_shape=data_shape,
            steps=steps,
        )

    print(f"Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        _infer()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"Timing ({n_iterations} iterations)...")
    times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        _infer()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_iterations}")

    return np.array(times)


def _print_stats(times, batch_size):
    tpv = times / batch_size
    print(f"\n{'='*60}")
    print("TIMING RESULTS")
    print(f"{'='*60}")
    print(f"\nBatch ({len(times)} iterations):")
    print(f"  Mean:   {times.mean():.4f}s  (±{times.std():.4f})")
    print(f"  Median: {np.median(times):.4f}s")
    print(f"  p95:    {np.percentile(times, 95):.4f}s")
    print(f"  p99:    {np.percentile(times, 99):.4f}s")
    print(f"\nPer-video (batch_size={batch_size}):")
    print(f"  Mean:       {tpv.mean():.4f}s  (±{tpv.std():.4f})")
    print(f"  Throughput: {1.0 / tpv.mean():.2f} videos/sec")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    cfg = _get_run_config(run_dir)
    dummy_ds = EchoDataset(cfg, split='test', n_missing_frames=args.n_missing_frames, cache=False)
    dummy_data = dummy_ds[0]

    model, ckpt_path = load_model_from_run(run_dir, dummy_data, args.ckpt_name)
    device = next(model.parameters()).device
    print(f"Loaded: {ckpt_path.name}  |  Device: {device}")

    batch = _create_batch(cfg, args.batch_size, args.n_missing_frames, device)
    data_shape = tuple(dummy_data['x'].shape)

    times = _benchmark(
        model=model,
        batch=batch,
        data_shape=data_shape,
        steps=args.steps,
        n_iterations=args.n_iterations,
        warmup=args.warmup,
    )

    _print_stats(times, args.batch_size)

    result_path = run_dir / "inference_timing.txt"
    with open(result_path, 'w') as f:
        f.write(f"Checkpoint:  {ckpt_path.name}\n")
        f.write(f"Device:      {device}\n")
        f.write(f"Batch size:  {args.batch_size}\n")
        f.write(f"Steps:       {args.steps}\n")
        f.write(f"Iterations:  {args.n_iterations}\n\n")
        f.write(f"Per-video timing:\n")
        f.write(f"  Mean:       {times.mean() / args.batch_size:.4f}s\n")
        f.write(f"  Std:        {times.std()  / args.batch_size:.4f}s\n")
        f.write(f"  Throughput: {args.batch_size / times.mean():.2f} videos/sec\n")
    print(f"Results saved to: {result_path}")


if __name__ == '__main__':
    main()
