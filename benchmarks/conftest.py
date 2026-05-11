"""
Shared pytest-benchmark fixtures for the EchoLVFM benchmark suite.

Two data/model regimes, toggled by `--lightweight`:

* `--lightweight` (default): tiny synthetic tensors + a tiny UNet3D. Fits on a
  12 GB GPU (or CPU) so the suite runs quickly on laptops.
* default: full-size UNet3D + EchoDataset from `paths=local` (sample_data or
  real CAMUS latents), for production-scale measurements on larger GPUs.

Override anything via CLI:

    pytest benchmarks/ --lightweight
    pytest benchmarks/ --rounds 10 --warmup 2
    pytest benchmarks/ --batch-size 2 --max-frames 16 --device cuda
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from dataset import default_eval_collate
from dataset.echodataset import EchoDataset
from src.flows import RMMFlow
from src.model import UNet3D


REPO_ROOT = Path(__file__).resolve().parents[1]
HYDRA_CONFIG_DIR = str(REPO_ROOT / "configs" / "flow_train")


# --- CLI options -----------------------------------------------------------

def pytest_addoption(parser):
    group = parser.getgroup("echolvfm benchmark")
    group.addoption("--lightweight", action="store_true",
                    help="Use tiny synthetic batch + tiny UNet3D (laptop-friendly).")
    group.addoption("--device", default=None,
                    help="Torch device override (e.g. 'cuda', 'cpu'). "
                         "Default: cuda if available, else cpu.")
    group.addoption("--batch-size", type=int, default=None,
                    help="Batch size override.")
    group.addoption("--max-frames", type=int, default=None,
                    help="Number of frames override.")
    group.addoption("--rounds", type=int, default=None,
                    help="pytest-benchmark rounds override.")
    group.addoption("--warmup", type=int, default=None,
                    help="pytest-benchmark warmup rounds override.")


# --- Config / regime -------------------------------------------------------

@dataclass
class BenchRegime:
    lightweight: bool
    device: torch.device
    batch_size: int
    max_frames: int
    latent_channels: int        # C (VAE latent channels)
    spatial: int                # H == W
    cross_attention_dim: int
    rounds: int
    warmup: int
    unet_kwargs: dict           # passed to UNet3D


def _pick_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def regime(request) -> BenchRegime:
    opts = request.config.option
    lightweight = bool(opts.lightweight)
    device = _pick_device(opts.device)

    if lightweight:
        # Tiny — head_dim=64 required because under pytest the JVPAttn triton
        # kernel hard-codes BLOCK_N=64 and asserts BLOCK_N <= HEAD_DIM.
        batch_size = opts.batch_size or 1
        max_frames = opts.max_frames or 8
        latent_channels = 4
        spatial = 16
        cross_attention_dim = 64
        unet_kwargs = dict(
            down_block_types=[
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ],
            up_block_types=[
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ],
            block_out_channels=[64, 128],
            num_attention_heads=[1, 2],         # head_dim = 64 in both blocks
            cross_attention_dim=cross_attention_dim,
            layers_per_block=1,
            transformer_layers_per_block=1,
        )
        # Dummy data has no "epoch" concept — just do a flat 100 steps.
        rounds = opts.rounds or 100
        warmup = opts.warmup if opts.warmup is not None else 3
    else:
        # Full — loads the real Hydra config and EchoDataset.
        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            cfg = compose(config_name="flow_train", overrides=["paths=local"])
        sample_ds = EchoDataset(cfg, split='train', cache=True)
        sample = sample_ds[0]
        C, T, H, W = sample['x'].shape
        latent_channels = C
        spatial = H
        cross_attention_dim = int(cfg.model.kwargs.get("cross_attention_dim", 1))
        batch_size = opts.batch_size or int(cfg.dataset.batch_size)
        max_frames = opts.max_frames or int(cfg.dataset.max_frames)
        unet_kwargs = dict(cfg.model.kwargs)
        # Sample enough steps to cover one epoch, but at least 100 for decent stats.
        steps_per_epoch = max(1, len(sample_ds) // batch_size)
        rounds = opts.rounds or max(steps_per_epoch, 100)
        warmup = opts.warmup if opts.warmup is not None else 3

    return BenchRegime(
        lightweight=lightweight,
        device=device,
        batch_size=batch_size,
        max_frames=max_frames,
        latent_channels=latent_channels,
        spatial=spatial,
        cross_attention_dim=cross_attention_dim,
        rounds=rounds,
        warmup=warmup,
        unet_kwargs=unet_kwargs,
    )


# --- Data ------------------------------------------------------------------

@pytest.fixture
def dummy_batch(regime: BenchRegime):
    """A training batch of synthetic latents with the right shape.

    Shape keys match what `RMMFlow.forward` / `LinearFlow.forward` expect:
      x:                      (B, C, T, H, W)
      cond_image:             (B, C+1, T, H, W)   (C masked latent + 1 pad mask)
      encoder_hidden_states:  (B, 1, cross_attn_dim)
      loss_mask:              (B, T)
    """
    B, C = regime.batch_size, regime.latent_channels
    T, H, W = regime.max_frames, regime.spatial, regime.spatial
    d = regime.device
    return {
        "x": torch.randn(B, C, T, H, W, device=d),
        "cond_image": torch.randn(B, C + 1, T, H, W, device=d),
        "encoder_hidden_states": torch.randn(B, 1, regime.cross_attention_dim, device=d),
        "loss_mask": torch.ones(B, T, device=d),
    }


@pytest.fixture
def real_batch(regime: BenchRegime):
    """A training batch drawn from EchoDataset (paths=local). Non-lightweight only."""
    if regime.lightweight:
        pytest.skip("real_batch is only meaningful without --lightweight")
    with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="flow_train", overrides=["paths=local"])
    ds = EchoDataset(cfg, split='train', cache=True)
    samples = [ds[i % len(ds)] for i in range(regime.batch_size)]
    batch = default_eval_collate(samples) if hasattr(default_eval_collate, "__call__") \
        else {k: torch.stack([s[k] for s in samples]) for k in samples[0]
              if isinstance(samples[0][k], torch.Tensor)}
    return {k: (v.to(regime.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


# --- Model / flow factories ------------------------------------------------

def _build_unet(regime: BenchRegime) -> UNet3D:
    C = regime.latent_channels
    return UNet3D(
        sample_size=regime.spatial,
        in_channels=2 * C + 1,    # noisy latent (C) + masked latent (C) + pad mask (1)
        out_channels=C,
        num_frames=regime.max_frames,
        **regime.unet_kwargs,
    )


@pytest.fixture
def rmmflow_factory(regime: BenchRegime):
    """Factory that returns a fresh `RMMFlow` on the benchmark device.

    The factory form lets each benchmark build a fresh model (important so
    `use_jvp_flash_attn` toggles the attention processor cleanly each time).
    """
    def _build(use_jvp_flash_attn: bool, **flow_kwargs) -> RMMFlow:
        unet = _build_unet(regime).to(regime.device)
        flow = RMMFlow(
            model=unet,
            use_jvp_flash_attn=use_jvp_flash_attn,
            # `prob_default_flow_obj=0.0` forces the JVP training path every step,
            # so both parametrizations exercise the code their name implies.
            prob_default_flow_obj=0.0,
            **flow_kwargs,
        ).to(regime.device)
        return flow
    return _build


# --- Memory accounting -----------------------------------------------------

@pytest.fixture(autouse=True)
def _cuda_clean():
    """Best-effort memory reset between benchmarks so peak-mem reads aren't
    contaminated by the previous test's allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def record_peak_memory(benchmark, device: torch.device) -> None:
    """Attach peak GPU memory (MiB since last reset) to the benchmark record.

    Call this AFTER the benchmarked callable has run its rounds. Works on CPU
    too (records 0.0 there — but the value still appears in JSON so the
    downstream summary script doesn't need to special-case the device).
    """
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
    else:
        peak_bytes = 0
    benchmark.extra_info["peak_gpu_mem_mib"] = peak_bytes / (1024 ** 2)
    benchmark.extra_info["device"] = str(device)
