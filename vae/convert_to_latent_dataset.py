import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from utils.util import select_device
from utils.video_utils import _is_video_file, load_video
from vae.util import load_vae_and_processor


def _ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def encode_video_to_latents(
    vae,
    processor,
    video_path: Path,
    out_path: Path,
    device: torch.device,
    save_dtype: torch.dtype = torch.float32,
    frame_shape = (112, 112)
):
    """Encode a single video into per-frame latent mu and var and save to out_path (.pt).

    Saved dict contains keys: 'mu', 'var'. mu/var are CPU tensors in `save_dtype`.
    """
    # load video as numpy (T, H, W, C)
    video = load_video(str(video_path), channels=3)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous().to(torch.uint8)

    # Prepare tensor for model: float32 in [0,1], then move to device
    video = video.to(device=device, dtype=save_dtype) / 255.0
    h, w = frame_shape
    video = processor.preprocess(video, height=h, width=w)
    with torch.no_grad():
        enc = vae.encode(video)
        mu = enc.latent_dist.mean
        var = enc.latent_dist.var

        # move to CPU and cast to save_dtype
        mu = mu.detach().cpu().to(save_dtype)
        var = var.detach().cpu().to(save_dtype)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mu": mu, "var": var}, out_path)


def convert_videos_dir_to_latents(
    repo_id: str,
    subfolder: str,
    input_dir: str,
    out_dir: str,
    device: Optional[torch.device] = None,
    ext: Optional[str] = "mp4",
    save_dtype: torch.dtype = torch.float32,
    max_videos: Optional[int] = None,
    skip_existing: bool = True,
    frame_shape = (112, 112)
):
    device = device or select_device()
    vae, processor = load_vae_and_processor(repo_id, subfolder, device)

    in_path = Path(input_dir)
    out_dir = f"{out_dir}_{subfolder.split('-')[-1]}"
    out_path = Path(out_dir)
    _ensure_outdir(out_path)

    pattern = "*" if ext is None else f"*.{ext}"
    files = sorted([p for p in in_path.rglob(pattern) if _is_video_file(p)])
    if max_videos is not None:
        files = files[:max_videos]

    for p in tqdm(files, desc="videos"):
        stem = p.stem
        target = out_path / f"{stem}.pt"
        if target.exists() and skip_existing:
            continue
        encode_video_to_latents(
            vae=vae,
            processor=processor,
            video_path=p,
            out_path=target,
            device=device,
            save_dtype=save_dtype,
            frame_shape=frame_shape
        )

    return out_path


def convert_videos(
    input_dir: str,
    out_dir: str,
    repo_id: str = "HReynaud/EchoFlow",
    subfolder: str = "vae/avae-4f4",
    ext: Optional[str] = "mp4",
    dtype: str = "float32",
    max_videos: Optional[int] = None,
    device: Optional[torch.device] = None,
    skip_existing: bool = True,
):
    """Convert videos in `input_dir` to latent mu/var files in `out_dir`.

    This is the programmatic API you can import and call from Python. Defaults use
    the cardiac VAE published at HReynaud/EchoFlow (subfolder vae/avae-4f4).
    """
    device = device or select_device()
    save_dtype = torch.float32 if dtype == "float32" else torch.float16

    return convert_videos_dir_to_latents(
        repo_id=repo_id,
        subfolder=subfolder,
        input_dir=input_dir,
        out_dir=out_dir,
        device=device,
        ext=ext,
        save_dtype=save_dtype,
        max_videos=max_videos,
        skip_existing=skip_existing,
    )

if __name__ == "__main__":
    input_dir = ""
    out_dir = ""
    convert_videos(
        subfolder="vae/avae-4f4",
        input_dir=input_dir,
        out_dir=out_dir,
    )
