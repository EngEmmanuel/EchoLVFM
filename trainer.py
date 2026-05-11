import torch
import hydra
import torch.optim as optim
from pathlib import Path
from datetime import datetime, timezone
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset.echodataset import EchoDataset
from dataset import default_eval_collate
from src.custom_callbacks import SampleAndCheckpointCallback
from utils.train import load_model, load_flow
from utils.util import select_device


class FlowVideoGenerator(LightningModule):
    def __init__(self, model, cfg, **kwargs):
        super().__init__()
        self.model = model
        self.cfg = cfg

        # Classifier-free guidance (CFG): register the learnable null EF embedding
        self.uncond_prob = cfg.trainer.get('uncond_prob', 0.0)
        if self.uncond_prob > 0.0:
            if not hasattr(self.model, 'null_ehs') or getattr(self.model, 'null_ehs') is None:
                self.model.register_parameter('null_ehs', torch.nn.Parameter(torch.zeros(1, 1)))
        else:
            if not hasattr(self.model, 'null_ehs'):
                setattr(self.model, 'null_ehs', None)

    def training_step(self, batch, batch_idx):
        batch = self.maybe_drop_cond(batch)
        out = self.model(**batch)
        loss = self._unwrap_and_log_loss(out, "train")
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        if getattr(self.model, 'null_ehs', None) is not None:
            self.log("train/null_ehs_value", float(self.model.null_ehs.detach().item()),
                on_step=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        loss = self._unwrap_and_log_loss(out, "val")
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        warmup_epochs = int(self.cfg.trainer.get("warmup_epochs", 0))
        start_factor = float(self.cfg.trainer.get("warmup_start_factor", 0.01))
        total_epochs = int(self.cfg.trainer.kwargs.max_epochs)
        lr = self.cfg.trainer.lr

        optimizer = optim.Adam(self.parameters(), lr=lr, **self.cfg.trainer.get('optim_kwargs', {}))

        match self.cfg.trainer.get('lr_scheduler', None):
            case None:
                if warmup_epochs > 0:
                    scheduler = LinearLR(
                        optimizer,
                        start_factor=start_factor,
                        end_factor=1.0,
                        total_iters=warmup_epochs
                    )
                    return [optimizer], [scheduler]
                return optimizer

            case 'cosineannealing':
                rem_epochs = max(1, total_epochs - warmup_epochs)
                main_scheduler = CosineAnnealingLR(optimizer, T_max=rem_epochs)

            case 'linear_decay':
                rem_epochs = max(1, total_epochs - warmup_epochs)
                def main_lr_lambda(epoch_in_main: int):
                    return max(0.0, float(rem_epochs - epoch_in_main) / float(rem_epochs))
                main_scheduler = LambdaLR(optimizer, lr_lambda=main_lr_lambda)

            case _:
                raise ValueError(f"Unsupported lr_scheduler: {self.cfg.trainer.lr_scheduler}")

        if warmup_epochs <= 0:
            scheduler = main_scheduler
        else:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            print(f"Warmup for {warmup_epochs} epochs, then "
                  f"{self.cfg.trainer.get('lr_scheduler', 'constant')} for "
                  f"{total_epochs - warmup_epochs} epochs.")
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )

        return [optimizer], [scheduler]

    def maybe_drop_cond(self, batch):
        """Classifier-free guidance: randomly replace EF embeddings with learned null token."""
        ehs = batch.get('encoder_hidden_states')
        if self.cfg.flow.type != 'linear' or self.uncond_prob <= 0.0:
            return batch

        B = ehs.shape[0]
        drop = (torch.rand(B, device=ehs.device) < self.uncond_prob)
        if drop.any():
            m = drop.view(B, 1)
            null = self.model.null_ehs.expand_as(ehs)
            ehs = torch.where(m, null, ehs)
            batch['encoder_hidden_states'] = ehs
        return batch

    def _unwrap_and_log_loss(self, out, split: str):
        """Accepts either a scalar loss or a dict with 'loss' key; logs components."""
        if isinstance(out, dict) and 'loss' in out:
            loss = out['loss']
            comps = {f"{split}_{k}": v for k, v in out.items() if k != 'loss'}
            if comps:
                self.log_dict(comps, prog_bar=False, on_step=True, on_epoch=True)
            return loss
        return out


@hydra.main(version_base=None, config_path="configs/flow_train", config_name="flow_train")
def main(cfg: DictConfig):
    device = select_device()

    # Setup output directories
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "sample_videos"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Datasets and DataLoaders
    train_ds = EchoDataset(cfg, split='train')
    val_ds = EchoDataset(cfg, split='val')
    sample_ds = EchoDataset(cfg, split='sample', n_missing_frames='max')

    train_dl = DataLoader(train_ds, batch_size=cfg.dataset.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.dataset.batch_size,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    sample_dl = DataLoader(sample_ds, batch_size=8, shuffle=False, collate_fn=default_eval_collate)
    dummy_data = train_ds[0]

    # Load model and flow wrapper
    model = load_model(cfg, dummy_data, device)
    model = load_flow(cfg, model)
    model = FlowVideoGenerator(model=model, cfg=cfg)

    # Callbacks
    callbacks_list = []
    for _, v in OmegaConf.to_container(cfg.ckpt, resolve=True).items():
        if isinstance(v, dict):
            callbacks_list.append(ModelCheckpoint(**v, dirpath=ckpt_dir))

    callbacks_list.append(LearningRateMonitor(logging_interval='step'))

    if cfg.get('sample'):
        callbacks_list.append(
            SampleAndCheckpointCallback(
                cfg=cfg,
                sample_dir=sample_dir,
                sample_dl=sample_dl,
                checkpoint_dir=ckpt_dir,
                device=device,
            )
        )

    # Logger
    config = OmegaConf.to_container(cfg, resolve=True)
    config.update({'local_output_dir': str(output_dir)})

    logger = None
    if cfg.get('wandb'):
        tags = list(cfg.wandb.get('tags', {}).values())
        logger = WandbLogger(
            **cfg.wandb.init_kwargs,
            tags=tags,
            save_dir=str(output_dir),
            config=config
        )
        if cfg.wandb.get('watch', False):
            logger.watch(model, log="all")

    # Train
    trainer = Trainer(logger=logger, callbacks=callbacks_list, **cfg.trainer.kwargs)

    utc_now = datetime.now(timezone.utc)
    print(utc_now.strftime("%A, %d %B %Y, %H:%M:%S %Z"))
    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    main()
