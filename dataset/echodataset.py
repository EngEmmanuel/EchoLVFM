import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from utils.util import select_device
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

device = select_device()

class EchoDataset(Dataset):
    def __init__(self, cfg, split='train', cache=True, **kwargs):
        self.cfg = cfg
        self.data_path = Path(cfg.dataset.path)
        self.df = pd.read_csv(self.data_path / 'metadata.csv')
        self.split = split.lower()
        self.cache = cache
        self._cache = {}  # dict for in-memory caching
        self.kwargs = kwargs

        # Filter dataframe based on split
        if split == 'sample':
            self.df = self.df[self.df['split'].str.lower() == 'val'].reset_index(drop=True)
        else:
            self.df = self.df[self.df['split'].str.lower() == split].reset_index(drop=True)

        # Sample subset for monitoring
        if kwargs.get('n_sample_videos', None) is not None:
            n_sample_videos = kwargs['n_sample_videos']
            self.df = self.df.sample(n=min(n_sample_videos, len(self.df))).reset_index(drop=True)

        # Maybe latent scaling
        if cfg.dataset.get('normalise_latents', False):
            scaling_path = Path(cfg.dataset.get('latent_scaling_path'))
            scaling = torch.load(scaling_path)
            self.mu_norm = scaling['mean']
            self.std_norm = scaling['std']

            print('Latent normalisation stats loaded:')
            print('mu_norm\n', self.mu_norm)
            print('std_norm\n', self.std_norm)

        self.ef_column = kwargs.get('ef_column', 'EF_AL')

        # Get data shape
        stats = torch.load(self.data_path / f"{self.df.iloc[0]['video_name']}.pt")
        self.shape = (self.cfg.dataset.max_frames,) + stats['mu'].shape[1:]

        print(f"{split} Video Shape: {self.shape}")  # (T, C, H, W)
        print(f"{split} Dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def resample_sequence(self, frames: torch.Tensor, target_length: int = 32):
        """
        Resample to fixed length. Returns resampled frames and a padding mask.
        """
        T, *data_shape = frames.shape
        if T < target_length:
            pad_shape = (target_length - T,) + frames.shape[1:]
            pad_block = torch.zeros(pad_shape, dtype=frames.dtype)
            resampled_frames = torch.cat([frames, pad_block], dim=0)
            pad_mask = torch.cat(
                [torch.zeros((T, 1, *data_shape[1:]), dtype=frames.dtype),
                 torch.ones((target_length - T, 1, *data_shape[1:]), dtype=frames.dtype)],
                dim=0
            )
        else:
            indices = torch.linspace(0, T - 1, target_length).round().long()
            resampled_frames = frames[indices]
            pad_mask = torch.zeros((target_length, 1, *data_shape[1:]), dtype=frames.dtype)

        return resampled_frames, pad_mask

    def mask_random_frames(self, video: torch.Tensor, dist='uniform'):
        '''
        Masks 'n_missing_frames' in the range [0, T) of the input video tensor.
        '''
        T = video.shape[0]

        # Allow specifying a fixed number or proportion of frames to mask
        if (self.split in ['train', 'val']):
            n_missing_frames = self.missing_frame_distribution(T, dist=dist)
            # Sample frame indices to mask
            mask_indices = random.sample(range(T), n_missing_frames)

        elif self.split == 'test' and ('every_' in self.kwargs.get('masking_distribution', '')):
            # Mask every n-th frame i.e. downsample the video by factor (n+1)x
            n = int(self.kwargs['masking_distribution'].split("_")[-1])
            keep_idx = list(range(0, T, n + 1))
            all_idx = set(range(T))
            mask_indices = sorted(all_idx - set(keep_idx))
        else: # also catches sample split
            n_missing_frames = self.kwargs.get('n_missing_frames')
            if isinstance(n_missing_frames, int): # if int, it's a fixed number of frames
                n_missing_frames = n_missing_frames
            if isinstance(n_missing_frames, float): # if 0 <= float < 1, it's a proportion of frames
                n_missing_frames = int(T * n_missing_frames)
            if n_missing_frames == 'max': # all but one frame
                n_missing_frames = T - 1
            assert 0 < n_missing_frames < T, "n_missing_frames must be in the range [0, T)"

            # Sample frame indices to mask
            mask_indices = random.sample(range(T), n_missing_frames)
            while (0 not in mask_indices) and (T-1 not in mask_indices): # ensure at least one of first or last frame is masked
                mask_indices = random.sample(range(T), n_missing_frames)


        mask = torch.ones_like(video)
        mask[mask_indices, ...] = 0
        masked_video = video * mask
        observed_mask = mask[:, 0:1, ...]

        return masked_video, observed_mask

    def missing_frame_distribution(self, T, dist):
        '''
        Returns the number of frames to remove based on the specified distribution.
        '''
        min_frames_removed = 1
        max_frames_removed = T - 1
        match dist:
            case 'uniform':
                val = np.random.randint(1, T)
            case 'geometric':
                p = 3.7/T
                val = T - np.random.geometric(p)
            case '1-frame':
                val = max_frames_removed

        return np.clip(val, min_frames_removed, max_frames_removed, dtype=int)

    def transform(self, z: torch.Tensor):
        # Random masking
        z_masked, observed_mask = self.mask_random_frames(
            video=z,
            dist=self.cfg.dataset.get('masking_distribution', 'uniform')
        )

        # Resample to fixed length
        resampled, pad_mask = self.resample_sequence(
            torch.cat([z, z_masked], dim=1),
            target_length=self.cfg.dataset.max_frames
        )
        z, cond = resampled[:, :self.shape[1], ...], resampled[:, self.shape[1]:, ...]

        if self.cfg.dataset.get('cond_pad_mask', False):
            cond = torch.cat([cond, (1. - pad_mask)], dim=1)

        outputs = {"z": z, "cond": cond}

        if self.split in ['sample', 'test']:
            observed_mask, _ = self.resample_sequence(observed_mask, target_length=self.cfg.dataset.max_frames)
            not_pad_mask = 1. - pad_mask[..., 0, 0, 0]

            outputs['observed_mask'] = observed_mask[..., 0, 0, 0]
            outputs['not_pad_mask'] = not_pad_mask
            return outputs


        # Mask for where to compute the loss. Only for training and validation
        match self.cfg.trainer.get('loss_mask'):
            case None:
                pass
            case 'pad_and_observed': # Generation only. Ignores all frames that are present in the (input AND condition).
                observed_mask, _ = self.resample_sequence(observed_mask, target_length=self.cfg.dataset.max_frames)
                #(T,C,H,W) -> (T,)
                outputs["loss_mask"] = observed_mask[..., 0, 0, 0]
            case 'pad': # Generation and Reconstruction. Ignores padded frames. Model has to learn to reconstruct seen frames
                outputs["loss_mask"] = 1. - pad_mask[..., 0, 0, 0]

        return outputs

    def process_ef(self, ef, dtype=None):
        ef = torch.tensor(ef, dtype=dtype)  # stay on CPU
        ehs_dim = self.cfg.model.kwargs.get('cross_attention_dim',
                                            self.cfg.model.kwargs.get('caption_channels'))
        encoder_hidden_states = ef.view(1, 1).expand(1, ehs_dim)  # shape [1, dim]
        return encoder_hidden_states

    def _load_video(self, video_name: str):
        '''
        Load latent stats from disk, with optional caching.
        '''
        if self.cache and video_name in self._cache:
            return self._cache[video_name]

        stats = torch.load(self.data_path / f"{video_name}.pt")
        if 'std' not in stats:
            stats['std'] = stats['var'].sqrt()
            stats.pop('var', None)

        if self.cache:
            self._cache[video_name] = stats
        return stats


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        stats = self._load_video(row['video_name'])
        mu, std = stats['mu'], stats['std']  # (T, C, H, W)

        eps = torch.zeros_like(mu)
        if self.cfg.dataset.get('add_eps_to_train_latents', True) and self.split == 'train':
            eps = torch.randn_like(mu)

        if self.cfg.dataset.get('normalise_latents', False):
            mu, std = self.normalise_latents(mu, std)
        
        z = (mu + std * eps) * self.cfg.vae.scaling_factor
        transformed_dict = self.transform(z)

        # Rearrange to (C, T, H, W), stay on CPU, ensure contiguous
        z = transformed_dict["z"].permute(1, 0, 2, 3).contiguous()
        cond = transformed_dict["cond"].permute(1, 0, 2, 3).contiguous()

        ef = row[self.ef_column] / 100.0
        encoder_hidden_states = self.process_ef(ef, dtype=z.dtype)

        inputs = {
            "x": z,
            "cond_image": cond,
            "encoder_hidden_states": encoder_hidden_states
        }

        # Potential additions
        if "loss_mask" in transformed_dict:
            inputs["loss_mask"] = transformed_dict["loss_mask"].contiguous()

        if self.split in ['sample', 'test']:
            inputs['video_name'] = row['video_name']
            inputs['observed_mask'] = transformed_dict.get('observed_mask')
            inputs['not_pad_mask'] = transformed_dict.get('not_pad_mask')

            if 'target_ef' in self.df.columns:
                inputs['target_ef_bin'] = row['target_ef_bin']

        
        return inputs


    def normalise_latents(self, mu: torch.Tensor, std: torch.Tensor):
        """
        Normalise per-channel using self.mu_norm/self.std_norm.
        Expects [T, C, H, W]. 
        """
        if not hasattr(self, 'mu_norm') or not hasattr(self, 'std_norm'):
            return (mu, std)
        mean_b = self.mu_norm.view(1, -1, 1, 1).to(device=mu.device, dtype=mu.dtype)
        std_b = self.std_norm.view(1, -1, 1, 1).to(device=std.device, dtype=std.dtype).clamp_min(1e-6)
        mu_n = (mu - mean_b) / std_b
        std_n = std / std_b
        return mu_n, std_n
