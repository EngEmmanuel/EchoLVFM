import time
import torch
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from utils.latent_utils import UnscaleLatents
from utils.train import SamplerConductor


class SampleAndCheckpointCallback(Callback):
    '''
    Callback to sample latents from the model and save checkpoints at specified
    intervals during training. Checkpoints saved here contain only model weights.
    '''
    def __init__(self, cfg, sample_dir: Path, sample_dl, checkpoint_dir: Path, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.sample_dir = sample_dir
        self.sample_dl = sample_dl
        self.checkpoint_dir = checkpoint_dir
        self._last_sample_epoch = 0
        self.device = device
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.conductor = SamplerConductor(cfg)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._sample_step(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        self._sample_step(trainer, pl_module, last=True)

    def _sample_step(self, trainer, pl_module, last=False):
        pl_module.model.to(self.device)

        if self.sample_dir is None or trainer.sanity_checking:
            return

        epoch = trainer.current_epoch

        is_sample_step = self.conductor.is_sample_step(
            epoch=epoch,
            last_sample_epoch=self._last_sample_epoch,
            last_step=last
        )

        if is_sample_step:
            if trainer.is_global_zero:
                out_name = 'last' if last else None
                sample_latents_from_model(
                    model=pl_module.model,
                    dl_list=[self.sample_dl],
                    run_cfg=self.cfg,
                    epoch=epoch,
                    step=trainer.global_step,
                    device=self.device,
                    samples_dir=self.sample_dir,
                    out_name=out_name
                )
                self._last_sample_epoch = epoch

            trainer.strategy.barrier()

            if not last:
                ckpt_name = f"sample-epoch={epoch}-step={trainer.global_step}.ckpt"
                trainer.save_checkpoint(
                    str(self.checkpoint_dir / ckpt_name),
                    weights_only=True
                )


def _make_gen_batch(batch) -> dict:
    """Create a generation batch by wrapping EF values."""
    wrapped_ef = ((batch['encoder_hidden_states'] + 0.5) % 1.0).clamp(0.15, 0.85)
    return {
        'cond_image': batch['cond_image'],
        'encoder_hidden_states': wrapped_ef
    }


def sample_latents_from_model(model, dl_list, run_cfg, epoch, step, device, samples_dir, out_name=None):
    """Sample from the model for each DataLoader and save latent videos with metadata."""
    _t0 = time.perf_counter()

    model.eval()
    C = int(run_cfg.vae.resolution.split('f')[0])

    model_sample_kwargs = run_cfg.sample.get('model_sample_kwargs', {})

    out_name = out_name or f"sample-epoch={epoch}-step={step}"
    samples_dir = Path(samples_dir) / out_name
    samples_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for dl in tqdm(dl_list, desc=f"Sampling: Epoch {epoch}"):
        unscale = UnscaleLatents(run_cfg, dl.dataset)
        data_shape = tuple(dl.dataset[0]['x'].shape)
        nmf = dl.dataset.kwargs.get('n_missing_frames', 'max')
        nmf = f"{int(100*nmf)}p" if isinstance(nmf, float) else str(nmf)

        for batch in tqdm(dl, desc="Batches"):
            reference_batch, input_batch = batch
            batch_size = input_batch['cond_image'].shape[0]

            sub_batches = {
                'rec': input_batch,
                'gen': _make_gen_batch(input_batch)
            }

            for tag, sub_batch in sub_batches.items():
                sub_batch = {k: v.to(device) for k, v in sub_batch.items()}

                videos = model.sample(
                    **sub_batch,
                    batch_size=batch_size,
                    data_shape=data_shape,
                    **model_sample_kwargs
                ).detach().cpu()
                videos = unscale(videos)

                for j, (ef, video) in enumerate(zip(sub_batch['encoder_hidden_states'][:, 0, 0].tolist(), videos)):
                    ef = round(int(100 * ef), 2)
                    real_name = reference_batch['video_name'][j]
                    video_name = f"{real_name}_ef{ef}_nmf{nmf}"

                    metadata_rows.append({
                        'video_name': video_name,
                        'n_missing_frames': nmf,
                        'EF': ef,
                        'rec_or_gen': tag,
                        'original_real_video_name': real_name,
                        'observed_mask': reference_batch['observed_mask'][j].tolist(),
                        'not_pad_mask': reference_batch['not_pad_mask'][j].tolist()
                    })
                    torch.save({'video': video}, samples_dir / f"{video_name}.pt")

    pd.DataFrame(metadata_rows).to_csv(samples_dir / 'metadata.csv', index=False)

    elapsed = time.perf_counter() - _t0
    print(f"Sampling done: epoch={epoch}, videos={len(metadata_rows)}, "
          f"time={elapsed/60:.1f}m, out='{samples_dir}'")
    return samples_dir
