"""
Training-step benchmarks for RMMFlow: `use_jvp_flash_attn` on vs off.

Run independently:
    pytest benchmarks/bench_training.py::test_rmmflow_training_step --lightweight
    pytest "benchmarks/bench_training.py::test_rmmflow_training_step[jvp-on]"

Run alongside everything else:
    pytest benchmarks/ --lightweight
"""
from __future__ import annotations

import pytest
import torch

from .conftest import record_peak_memory


@pytest.mark.parametrize("use_jvp", [True, False], ids=["jvp-on", "jvp-off"])
def test_rmmflow_training_step(benchmark, regime, rmmflow_factory, dummy_batch, use_jvp):
    """One forward + backward + optimizer step of RMMFlow.

    Parametrized on `use_jvp_flash_attn`. Peak GPU memory for the whole
    benchmark run (all rounds) is attached to `benchmark.extra_info` so the
    summariser can show it alongside the timing stats.
    """
    # JVPAttn requires CUDA (triton kernels). On CPU the jvp-on parametrization
    # can't run; skip with a clear reason so the summary shows why.
    if use_jvp and regime.device.type != "cuda":
        pytest.skip("use_jvp_flash_attn=True requires a CUDA device (triton kernels).")

    flow = rmmflow_factory(use_jvp_flash_attn=use_jvp)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
    batch = dummy_batch

    def _one_step():
        optimizer.zero_grad(set_to_none=True)
        out = flow(**batch)
        loss = out["loss"] if isinstance(out, dict) else out
        loss.backward()
        optimizer.step()
        if regime.device.type == "cuda":
            torch.cuda.synchronize()

    benchmark.pedantic(
        _one_step,
        rounds=regime.rounds,
        warmup_rounds=regime.warmup,
        iterations=1,
    )

    benchmark.extra_info["axis"] = "use_jvp_flash_attn"
    benchmark.extra_info["use_jvp_flash_attn"] = bool(use_jvp)
    benchmark.extra_info["batch_size"] = regime.batch_size
    benchmark.extra_info["max_frames"] = regime.max_frames
    benchmark.extra_info["latent_channels"] = regime.latent_channels
    benchmark.extra_info["spatial"] = regime.spatial
    benchmark.extra_info["cross_attention_dim"] = regime.cross_attention_dim
    benchmark.extra_info["lightweight"] = regime.lightweight
    benchmark.extra_info["rounds_configured"] = regime.rounds
    benchmark.extra_info["warmup_configured"] = regime.warmup
    benchmark.extra_info["unet_kwargs"] = {
        k: (list(v) if isinstance(v, (list, tuple)) else v)
        for k, v in regime.unet_kwargs.items()
    }
    record_peak_memory(benchmark, regime.device)
