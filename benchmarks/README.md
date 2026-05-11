# EchoLVFM benchmarks

Pytest-based benchmark suite measuring RMMFlow training-step speed and peak
GPU memory with `use_jvp_flash_attn` on vs off.

## Two regimes

| Regime | When | Data | Model |
|--------|------|------|-------|
| `--lightweight` | laptops / 12 GB boxes | synthetic dummy tensors | tiny UNet3D (block_out=[64,128], head_dim=64) |
| default | production GPUs | `EchoDataset` via `paths=local` | full UNet3D from `configs/flow_train/model/unet.yaml` |

`--lightweight` avoids loading the real dataset and keeps memory tiny enough
to run on CPU or a small GPU. Default regime uses whatever Hydra composes from
`configs/flow_train/flow_train.yaml` with `paths=local` — i.e. the same config
`trainer.py` trains with.

## Quickstart

```bash
# Lightweight smoke (synthetic data, tiny UNet)
python benchmarks/run.py --lightweight

# Full benchmark with real CAMUS latents
python benchmarks/run.py
```

Outputs land in `benchmarks/results/<timestamp>/`:

- `config.json`   — the model + training config this run used (self-documenting)
- `benchmark.json` — raw pytest-benchmark stats (slimmed — cpuinfo stripped)
- `summary.md`    — per-benchmark table
- `time.png`      — mean step time, error bars = stddev across rounds
- `peak_mem.png`  — peak GPU memory per step

## Running with real data

`paths=local` is the Hydra override the default regime uses. So:

1. Edit `configs/flow_train/paths/local.yaml` so `dataset.root` / `dataset.metadata_file`
   point at your CAMUS latents + metadata CSV.
2. Run `python benchmarks/run.py` (no `--lightweight` flag).

By default `rounds = max(len(dataset) // batch_size, 100)` — enough to cover
one training epoch's worth of distinct batches, with a floor of 100 samples
for decent statistics. Override with `--rounds N`.

## Running individual benchmarks

Because everything is pytest, you can bypass `run.py` and invoke pytest
directly — useful for iterating on one parametrization or filtering:

```bash
# Just the jvp-on case
pytest "benchmarks/bench_training.py::test_rmmflow_training_step[jvp-on]" --lightweight

# Both cases, pytest-benchmark's own output only (no plots)
pytest benchmarks/bench_training.py --lightweight

# Filter by ID keyword
pytest benchmarks/ --lightweight -k jvp-off
```

To keep pytest output AND get plots, pass pytest args through `run.py` after
`--`:

```bash
python benchmarks/run.py --lightweight -- -k jvp-on -s
```

## Overriding configuration

All knobs take precedence in this order: CLI flag > config file > regime default.

| Flag | Effect |
|------|--------|
| `--device cuda` / `--device cpu` | Torch device (default: cuda if available) |
| `--batch-size N` | Override batch size |
| `--max-frames N` | Override number of frames per clip |
| `--rounds N` | Override pytest-benchmark rounds (measured samples) |
| `--warmup N` | Override warmup rounds |
| `--results-dir PATH` | Custom output folder (default: `benchmarks/results/<ts>/`) |

Examples:

```bash
# Production GPU, force batch size 4, 200 measurement rounds
python benchmarks/run.py --batch-size 4 --rounds 200

# Memory-starved: tiny shapes but keep the real-data code path
python benchmarks/run.py --batch-size 1 --max-frames 4

# CPU-only (skips the jvp-on parametrization — JVPAttn needs CUDA/triton)
python benchmarks/run.py --lightweight --device cpu
```

## Requirements for the JVP path

`jvp-flash-attention` requires a CUDA GPU and `triton`. On Windows install
`triton-windows<3.5` (see the main README for the version matrix). The
CPU fallback parametrization (`jvp-off`) runs everywhere.

One non-obvious constraint: under pytest, the JVPAttn kernel hard-codes
`BLOCK_N=64` and asserts `BLOCK_N <= HEAD_DIM`. That's why the lightweight
regime uses `block_out_channels=[64, 128]` with `num_attention_heads=[1, 2]`
(→ `head_dim = 64`) instead of something smaller. If you pick your own
lightweight config via overrides, make sure the attention head dimension is
≥ 64.

## What gets measured

Each benchmark runs `forward + backward + optimizer.step + cuda.synchronize`
as one timed step. Stats (mean, median, stddev) are aggregated across
`rounds` samples; peak GPU memory is the high-water mark across the whole
run, measured via `torch.cuda.max_memory_allocated()` after a fresh
`reset_peak_memory_stats()`. See `conftest.py::record_peak_memory` and the
`_cuda_clean` autouse fixture for details.
