import torch
import pandas as pd

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from utils.util import select_device
from utils.train import load_model, load_flow


def _find_checkpoint(ckpt_dir: Path, ckpt_name: Optional[str] = None) -> Path:
    '''
    Returns the path to a checkpoint file. If no specific
    checkpoint is requested, returns the latest.
    '''
    if ckpt_name:
        p = ckpt_dir / ckpt_name
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    candidates = list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _get_run_config(run_dir: Path) -> OmegaConf:
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {hydra_cfg_path}")
    return OmegaConf.load(hydra_cfg_path)


def load_model_from_run(run_dir: str | Path, dummy_data: dict, ckpt_name: Optional[str] = None):
    """Load config and checkpoint weights for a trained run.

    Args:
        run_dir:   Path to a Hydra run directory containing '.hydra/config.yaml'
                   and 'checkpoints/*.ckpt'.
        dummy_data: A sample from the dataset (used to infer input shapes for model init).
        ckpt_name: Optional checkpoint filename; defaults to last.ckpt or the newest .ckpt.

    Returns:
        model:     Flow-wrapped model with weights loaded and moved to device.
        ckpt_path: Resolved Path to the checkpoint used.
    """
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)

    cfg = _get_run_config(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found at {ckpt_dir}")

    device = select_device()
    base_model = load_model(cfg, dummy_data, device)
    model = load_flow(cfg, base_model).to(device)

    ckpt_path = _find_checkpoint(ckpt_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # FlowVideoGenerator is saved as a LightningModule with attribute 'model'.
    # Strip leading 'model.' prefix to match the flow wrapper keys.
    cleaned = {
        k.split("model.", 1)[1] if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }

    # Normalise null embedding key to 'null_ehs' for backward compatibility.
    null_keys = [k for k in list(cleaned.keys()) if k == 'null_ehs' or k.endswith('.null_ehs')]
    null_tensor = None
    for k in null_keys:
        if k == 'null_ehs':
            null_tensor = cleaned[k]
            break
    if null_tensor is None and null_keys:
        null_tensor = cleaned[null_keys[0]]
        cleaned['null_ehs'] = null_tensor
    for k in null_keys:
        if k != 'null_ehs' and k in cleaned:
            cleaned.pop(k)

    if null_tensor is not None:
        if not hasattr(model, 'null_ehs') or not isinstance(getattr(model, 'null_ehs'), torch.nn.Parameter):
            model.register_parameter('null_ehs', torch.nn.Parameter(null_tensor.clone()))
        else:
            with torch.no_grad():
                model.null_ehs.copy_(null_tensor)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[load_model_from_run] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[load_model_from_run] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.eval()
    return model, ckpt_path
