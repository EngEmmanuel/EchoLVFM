import torch
from utils.util import _ensure_broadcast


class UnscaleLatents:
    """
    Converts model-output latents back to raw VAE latent space.

    Reverses the VAE scaling factor and optional per-channel normalisation
    applied during dataset preparation, making the output suitable for
    vae.decode().

    Args:
        run_cfg: Hydra config containing vae.scaling_factor and
                 dataset.normalise_latents.
        dataset:  Dataset instance; must expose mu_norm and std_norm (1-D
                  tensors) when normalise_latents is True.
    """

    def __init__(self, run_cfg, dataset):
        self.vae_scaling_factor = run_cfg.vae.scaling_factor
        self.mu_norm = None
        self.std_norm = None

        if run_cfg.dataset.get('normalise_latents', False):
            assert hasattr(dataset, 'mu_norm') and hasattr(dataset, 'std_norm'), \
                "normalise_latents=True but dataset is missing mu_norm / std_norm."
            assert dataset.mu_norm is not None and dataset.std_norm is not None, \
                "normalise_latents=True but dataset.mu_norm or std_norm is None."
            self.mu_norm = dataset.mu_norm
            self.std_norm = dataset.std_norm.clamp_min(1e-6)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.vae_scaling_factor
        if self.mu_norm is None or self.std_norm is None:
            return z
        mu_b = _ensure_broadcast(self.mu_norm, z)
        std_b = _ensure_broadcast(self.std_norm, z)
        return z * std_b + mu_b
