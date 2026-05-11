"""Load EchoLVFM checkpoints from a Hugging Face Hub model repo.

Mirrors the `vae/util.py::load_vae_and_processor` shape: one function call
returns a built, ready-to-use flow object. Space-compatible — no local
filesystem assumptions beyond the HF cache.
"""
from __future__ import annotations

from typing import Union

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

from src.flows import LinearFlow, RMMFlow
from src.model import UNet3D


Flow = Union[RMMFlow, LinearFlow]


def load_model_from_hub(
    repo_id: str,
    subfolder: str,
    device: Union[str, torch.device] = "cuda",
    revision: str | None = None,
) -> Flow:
    """Download one variant from a Hub repo and return the built flow object.

    Only the requested subfolder's `config.yaml` + `model.safetensors` are
    fetched — sibling subfolders aren't touched.

    Args:
        repo_id:    e.g. "EngEmmanuel/EchoLVFM-Weights".
        subfolder:  "mean_h1", "mean_h2", or "linear".
        device:     torch device for the built flow.
        revision:   optional git revision (branch/tag/commit).

    Returns:
        A `RMMFlow` or `LinearFlow` with weights loaded, moved to `device`,
        in eval mode.
    """
    cfg_path = hf_hub_download(repo_id, "config.yaml",
                               subfolder=subfolder, revision=revision)
    weights_path = hf_hub_download(repo_id, "model.safetensors",
                                   subfolder=subfolder, revision=revision)

    cfg = OmegaConf.load(cfg_path)

    if cfg.model.type.lower() != "unet":
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
    unet = UNet3D(
        sample_size=cfg.model.init.sample_size,
        in_channels=cfg.model.init.in_channels,
        out_channels=cfg.model.init.out_channels,
        num_frames=cfg.model.init.num_frames,
        **OmegaConf.to_container(cfg.model.kwargs, resolve=True),
    )

    flow_kwargs = OmegaConf.to_container(cfg.flow.get("kwargs", {}), resolve=True)
    flow_type = cfg.flow.type.lower()
    if flow_type == "mean":
        flow: Flow = RMMFlow(model=unet, **flow_kwargs)
    elif flow_type == "linear":
        flow = LinearFlow(model=unet, **flow_kwargs)
    else:
        raise ValueError(f"Unsupported flow type: {cfg.flow.type}")

    state = load_file(weights_path)
    # Strict: we control the upload format, so any key mismatch means the
    # artefact is malformed — surface it loudly rather than limping along.
    flow.load_state_dict(state, strict=True)

    flow.to(device).eval()
    return flow
