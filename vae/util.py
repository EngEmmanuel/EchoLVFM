
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from pathlib import Path
from typing import Optional, List


def load_vae_and_processor(vae_locator: str, subfolder: Optional[str], device: torch.device):
    """Load AutoencoderKL from a local folder or a HuggingFace repo id and return (vae, processor).

    vae_locator: local path or HF repo id (author/repo)
    subfolder: optional subfolder inside HF repo where VAE lives
    device: torch device
    """
    p = Path(vae_locator)
    if p.exists():
        print(f"Loading local VAE from: {vae_locator}")
        vae = AutoencoderKL.from_pretrained(str(p))
    else:
        print(f"Loading VAE from HuggingFace repo: {vae_locator}, subfolder={subfolder}")
        vae = AutoencoderKL.from_pretrained(vae_locator, subfolder=subfolder)

    vae = vae.to(device)
    vae.eval()

    processor = VaeImageProcessor.from_config(vae.config)
    print("Created VaeImageProcessor from config")

    return vae, processor