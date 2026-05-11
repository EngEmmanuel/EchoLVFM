"""
Repackage EchoLVFM Lightning/weights-only checkpoints into a clean
`<stage_dir>/<subfolder>/{model.safetensors, config.yaml}` layout, then (by
default) upload to a Hugging Face model repo.

Layout produced:

    <stage_dir>/
      echolvfm_h1/
        model.safetensors
        config.yaml
      echolvfm_h2/
        model.safetensors
        config.yaml
      linear/
        model.safetensors
        config.yaml

Each `model.safetensors` is the flow-level state dict (leading `model.`
prefix from `FlowVideoGenerator` stripped). Each `config.yaml` contains only
the fields a loader needs to rebuild the flow — no dataset paths or trainer
settings leak to the public repo.

Examples:

    # Stage only (dry-run — produces files, skips upload). Good for inspecting.
    python scripts/upload_to_hub.py --stage-only

    # Stage + upload to the default private repo.
    python scripts/upload_to_hub.py

    # Stage + upload to a different repo.
    python scripts/upload_to_hub.py --repo-id SomeUser/SomeRepo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]

# Each local checkpoint → target subfolder inside the Hub repo.
# `hydra_cfg` sits next to the checkpoint under `.hydra/config.yaml` for every
# run in this repo.
CHECKPOINTS = [
    {
        "ckpt":      REPO_ROOT / "ckpts/echolvfm_h=1/last.ckpt",
        "hydra_cfg": REPO_ROOT / "ckpts/echolvfm_h=1/.hydra/config.yaml",
        "subfolder": "echolvfm_h1",
    },
    {
        "ckpt":      REPO_ROOT / "ckpts/echolvfm_h=2/sample-epoch=300-step=120400.ckpt",
        "hydra_cfg": REPO_ROOT / "ckpts/echolvfm_h=2/.hydra/config.yaml",
        "subfolder": "echolvfm_h2",
    },
    {
        "ckpt":      REPO_ROOT / "ckpts/linear/sample-epoch=200-step=80400.ckpt",
        "hydra_cfg": REPO_ROOT / "ckpts/linear/.hydra/config.yaml",
        "subfolder": "linear",
    },
]

DEFAULT_REPO_ID = "EngEmmanuel/EchoLVFM-Weights"


def _flatten_state_dict(raw: dict) -> dict:
    """Undo the `FlowVideoGenerator` Lightning wrapper so keys sit at flow level.

    Mirrors `evaluation/functions.py::load_model_from_run` (the local analogue).
    """
    state = raw.get("state_dict", raw)
    cleaned = {
        (k.split("model.", 1)[1] if k.startswith("model.") else k): v
        for k, v in state.items()
    }
    # No checkpoint in this batch was trained with CFG, so `null_ehs` should be
    # absent. Warn loudly if one slips through — skipping it silently would
    # lose a parameter the loader might need.
    null_keys = [k for k in cleaned if k == "null_ehs" or k.endswith(".null_ehs")]
    if null_keys:
        print(f"  [WARN] found null_ehs keys {null_keys!r} — this batch was "
              f"supposed to be uncond_prob=0. Leaving them in the saved weights.")
    return cleaned


def _infer_channels(state: dict) -> tuple[int, int]:
    """Return (in_channels, out_channels) read from the UNet's input/output convs.

    Keys are flow-level after stripping the Lightning wrapper, so the UNet
    sits under `model.` — `model.conv_in.weight`, `model.conv_out.weight`.
    """
    conv_in = state.get("model.conv_in.weight")
    conv_out = state.get("model.conv_out.weight")
    if conv_in is None or conv_out is None:
        raise KeyError("state_dict missing model.conv_in.weight or "
                       "model.conv_out.weight — prefix stripping may have failed.")
    return int(conv_in.shape[1]), int(conv_out.shape[0])


def _infer_sample_size(cfg: DictConfig, hydra_cfg_path: Path) -> int:
    """Spatial size of a latent, in VAE space.

    `sample_size` is metadata-only on UNetSpatioTemporalConditionModel (config
    field, not used in forward), but we keep it accurate for documentation.
    Read it from one real latent — `sample_data/CAMUS_Latents_<res>/` is
    bundled with the repo for this purpose, so this works without a full
    CAMUS install.
    """
    resolution = str(cfg.vae.resolution)
    data_path = REPO_ROOT / "sample_data" / f"CAMUS_Latents_{resolution}"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected bundled latents at {data_path} to infer sample_size — "
            f"not found. hydra cfg: {hydra_cfg_path}"
        )
    meta = data_path / "metadata.csv"
    first_name = meta.read_text().splitlines()[1].split(",")[0]
    first_pt = data_path / f"{first_name}.pt"
    sample = torch.load(first_pt, map_location="cpu")
    mu = sample["mu"]  # (T, C, H, W)
    h, w = int(mu.shape[-2]), int(mu.shape[-1])
    if h != w:
        raise ValueError(f"Non-square latent ({h}x{w}); UNet3D assumes square.")
    return h


def repackage_one(ckpt_path: Path, hydra_cfg_path: Path, subfolder: str,
                  stage_dir: Path) -> None:
    print(f"\n-- {subfolder} --")
    print(f"  ckpt:   {ckpt_path}")
    print(f"  hydra:  {hydra_cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(hydra_cfg_path)

    cfg = OmegaConf.load(hydra_cfg_path)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _flatten_state_dict(raw)
    in_ch, out_ch = _infer_channels(state)
    sample_size = _infer_sample_size(cfg, hydra_cfg_path)
    num_frames = int(cfg.dataset.max_frames)

    out_dir = stage_dir / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "model.safetensors"
    # save_file requires contiguous tensors with no shared storage.
    state = {k: v.detach().contiguous().clone() for k, v in state.items()}
    save_file(state, str(weights_path))
    size_gb = weights_path.stat().st_size / (1024**3)
    print(f"  wrote:  {weights_path.relative_to(stage_dir)}  ({size_gb:.1f} GB, "
          f"{len(state)} tensors)")

    # Minimal config: only what the loader needs to rebuild the flow + UNet.
    # Explicitly NOT saving dataset/paths/trainer — they're local and irrelevant
    # to inference.
    pub_cfg = OmegaConf.create({
        "model": {
            "type": cfg.model.type,
            "kwargs": OmegaConf.to_container(cfg.model.kwargs, resolve=True),
            "init": {
                "sample_size": sample_size,
                "in_channels": in_ch,
                "out_channels": out_ch,
                "num_frames": num_frames,
            },
        },
        "flow": {
            "type": cfg.flow.type,
            "kwargs": OmegaConf.to_container(cfg.flow.get("kwargs", {}), resolve=True),
            "sample_kwargs": OmegaConf.to_container(
                cfg.flow.get("sample_kwargs", {}), resolve=True),
        },
    })
    cfg_path = out_dir / "config.yaml"
    OmegaConf.save(pub_cfg, cfg_path)
    print(f"  wrote:  {cfg_path.relative_to(stage_dir)}  "
          f"(flow={cfg.flow.type}, in={in_ch}, out={out_ch}, "
          f"T={num_frames}, sample_size={sample_size})")


def _hf_upload(stage_dir: Path, repo_id: str, private: bool) -> int:
    create_cmd = ["hf", "repos", "create", repo_id, "--type", "model", "--exist-ok"]
    if private:
        create_cmd.append("--private")
    print("\n$", " ".join(create_cmd))
    rc = subprocess.call(create_cmd)
    if rc != 0:
        return rc

    upload_cmd = [
        "hf", "upload", repo_id, str(stage_dir), ".",
        "--type", "model",
        "--commit-message", "Initial checkpoints: echolvfm_h1, echolvfm_h2, linear",
    ]
    print("$", " ".join(upload_cmd))
    return subprocess.call(upload_cmd)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-only", action="store_true",
                   help="Produce staged files and stop — no upload.")
    p.add_argument("--stage-dir", default=None,
                   help="Where to write staged files. Default: a tempdir.")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"Target Hub repo id (default: {DEFAULT_REPO_ID}).")
    p.add_argument("--public", action="store_true",
                   help="Create the repo as public (default: private).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.stage_dir:
        stage_dir = Path(args.stage_dir).resolve()
        stage_dir.mkdir(parents=True, exist_ok=True)
        using_tempdir = False
    else:
        stage_dir = Path(tempfile.mkdtemp(prefix="echolvfm_stage_"))
        using_tempdir = True
    print(f"Stage dir: {stage_dir}")

    for entry in CHECKPOINTS:
        repackage_one(
            ckpt_path=entry["ckpt"],
            hydra_cfg_path=entry["hydra_cfg"],
            subfolder=entry["subfolder"],
            stage_dir=stage_dir,
        )

    if args.stage_only:
        print(f"\nStaged files at: {stage_dir}")
        print("Inspect, then re-run without --stage-only to upload.")
        return 0

    rc = _hf_upload(stage_dir, args.repo_id, private=not args.public)
    if rc == 0:
        print(f"\nUploaded to https://huggingface.co/{args.repo_id}")
        if using_tempdir:
            print(f"(Tempdir {stage_dir} left in place — delete manually if you "
                  f"don't need it.)")
    else:
        print(f"\nUpload failed (rc={rc}). Stage dir preserved at: {stage_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
