"""
Compute per-channel latent mean/std (scaling.pt) from saved VAE latents.

Each video file is expected to be a torch .pt dict with keys:
  - 'mu' : Tensor (T, C, H, W)
  - either 'std' or 'var' : Tensor (T, C, H, W)

metadata.csv must have columns: 'video_name' and 'split' where split in {TRAIN, VAL, TEST}.

Usage:
  python normalise_latent_dataset.py --data-dir /path/to/CAMUS_Latents_4f4 \
                                     --metadata /path/to/CAMUS_Latents_4f4/metadata.csv \
                                     --out scaling_mydata.pt
  # optionally compute stats on sampled z instead of mu:
  python normalise_latent_dataset.py ... --use-sampled
"""
from pathlib import Path
import argparse
import torch
import pandas as pd
from tqdm import tqdm

def tensor_to_channel_stats(t: torch.Tensor):
    """
    Given a tensor of shape (..., C, H, W) or (C, H, W) returns:
    sum_per_channel, sumsq_per_channel, n_elements_per_channel
    where sums are over all non-channel dims.
    Returned shapes: (C,), (C,), scalar or (C,) count (we'll use scalar count per channel).
    """
    # ensure shape (..., C, H, W). #confirmed
    if t.ndim == 3:
        # (C, H, W) -> add batch dim
        t = t.unsqueeze(0)
    # collapse all dims except channel
    # target shape (N, C, H, W) -> (N * H * W, C)
    C = t.shape[1]
    flattened = t.permute(1, 0, 2, 3).reshape(C, -1)  # (C, N*H*W)
    sum_per_channel = flattened.sum(dim=1)
    sumsq_per_channel = (flattened * flattened).sum(dim=1)
    n = flattened.shape[1]  # number of values per channel in this tensor
    return sum_per_channel, sumsq_per_channel, n

def main(args):
    data_dir = Path(args.data_dir)
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {meta_path}")

    df = pd.read_csv(meta_path)
    # filter to TRAIN and VAL only
    df = df[df['split'].str.upper().isin({'TRAIN', 'VAL'})]
    video_names = df['video_name'].unique().tolist()
    if not video_names:
        raise RuntimeError("No TRAIN/VAL video entries found in metadata.csv")

    sum_ch = None
    sumsq_ch = None
    total_count = 0
    channels = None

    for v in tqdm(video_names, desc="Processing videos"):
        p = data_dir / f"{v}.pt"
        if not p.exists():
            # skip missing with a warning
            tqdm.write(f"Warning: file not found {p}, skipping")
            continue

        stats = torch.load(p)
        if 'mu' not in stats:
            tqdm.write(f"Warning: 'mu' not in {p}, skipping")
            continue

        mu = stats['mu'].float()  # (T, C, H, W) or (C, H, W)
        # get std
        if 'std' in stats:
            std = stats['std'].float()
        elif 'var' in stats:
            std = stats['var'].float().sqrt()
        else:
            tqdm.write(f"Warning: neither 'std' nor 'var' found in {p}; assuming zeros")
            std = torch.zeros_like(mu)

        # decide whether to use sampled z or mu
        if args.use_sampled:
            # sample one eps per frame
            eps = torch.randn_like(mu)
            z = mu + std * eps
        else:
            # use deterministic mu
            z = mu

        # accumulate per-channel sums
        s, ssq, n = tensor_to_channel_stats(z)
        if sum_ch is None:
            sum_ch = s
            sumsq_ch = ssq
            total_count = n
            channels = s.shape[0]
        else:
            if s.shape[0] != channels:
                raise ValueError(f"Channel mismatch in {p}: expected {channels}, got {s.shape[0]}")
            sum_ch = sum_ch + s
            sumsq_ch = sumsq_ch + ssq
            total_count += n

    if sum_ch is None:
        raise RuntimeError("No valid videos processed; aborting.")

    # compute mean and std per channel
    mean = sum_ch / total_count            # shape (C,)
    var = (sumsq_ch / total_count) - (mean * mean)
    # numerical safety
    var = var.clamp(min=0.0)
    std = var.sqrt()

    out = {'mean': mean, 'std': std}
    out_path = Path(args.out).resolve()
    torch.save(out, out_path)
    print(f"Saved scaling to {out_path}")
    print("mean shape:", mean.shape, "std shape:", std.shape)
    print("example mean:", mean)
    print("example std:", std)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute scaling.pt from directory of latent videos")
    p.add_argument("--data-dir", required=True, help="Directory containing video .pt latent files")
    p.add_argument("--metadata", required=True, help="Path to metadata.csv with video_name and split columns")
    p.add_argument("--out", default="scaling.pt", help="Output filename for scaling (default: scaling.pt)")
    p.add_argument("--use-sampled", action="store_true",
                   help="Compute stats on sampled z = mu + std * eps instead of mu (default: mu)")
    args = p.parse_args()
    main(args)
