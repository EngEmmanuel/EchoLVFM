from src.model import UNet3D
from src.flows import LinearFlow, RMMFlow
from vae.util import load_vae_and_processor


def load_model(cfg, dummy_data, device):
    if isinstance(dummy_data, dict):
        C, T, H, W = dummy_data['x'].shape
        Cc, _, _, _ = dummy_data['cond_image'].shape

    print(f'Input shape: {(C+Cc, T, H, W)}, with {Cc} cond channels')

    match cfg.model.type.lower():
        case "unet":
            return UNet3D(
                sample_size=W,
                in_channels=C + Cc,
                out_channels=C,
                num_frames=T,
                **cfg.model.kwargs
            ).to(device)

    raise ValueError(f"Unsupported model type: {cfg.model.type}")


def load_flow(cfg, model):
    match cfg.flow.type.lower():
        case "linear":
            return LinearFlow(model=model, **cfg.flow.get('kwargs', {}))
        case "mean":
            return RMMFlow(model=model, **cfg.flow.get('kwargs', {}))

    raise ValueError(f"Unsupported flow type: {cfg.flow.type}")


def load_vae_processor(cfg, device):
    return load_vae_and_processor(
        vae_locator=cfg.vae.repo_id,
        subfolder=cfg.vae.get('subfolder', None),
        device=device
    )


class SamplerConductor:
    """Controls how often the model is sampled during training."""

    def __init__(self, run_cfg):
        self.max_epochs = run_cfg.trainer.kwargs.max_epochs
        self.sample_every_n_epochs = run_cfg.sample.get('every_n_epochs', self.max_epochs)
        self.scheduler = run_cfg.trainer.get('lr_scheduler', None)

    def is_sample_step(self, epoch, last_sample_epoch, last_step):
        if last_step:
            return True

        epoch_freq = self.sample_every_n_epochs

        # Sample more frequently near the cosine annealing transition region
        if self.scheduler == 'cosineannealing':
            progress = epoch / self.max_epochs
            roi = (0.4, 0.55)
            if roi[0] <= progress <= roi[1]:
                epoch_freq = max(1, self.sample_every_n_epochs // 2)

        return epoch % epoch_freq == 0 and epoch != last_sample_epoch
