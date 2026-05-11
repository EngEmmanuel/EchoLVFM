import torch 

def select_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_broadcast(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast per-channel (C,) tensor to ref:
       ref ndim 4 -> shape (C,1,1,1) for [C,T,H,W]
       ref ndim 5 -> shape (1,C,1,1,1) for [B,C,T,H,W]
    """
    if t.ndim != 1:
        raise ValueError("t must be 1D of shape (C,)")
    if ref.ndim not in (4, 5):
        raise ValueError("ref must have ndim 4 ([C,T,H,W]) or 5 ([B,C,T,H,W])")

    C = t.shape[0]
    if ref.ndim == 4:
        if ref.shape[0] != C:
            raise ValueError(f"Channel mismatch: t has C={C}, ref[0]={ref.shape[0]}")
        b = t.view(C, 1, 1, 1)            # (C,1,1,1)
    else:  # ndim == 5
        if ref.shape[1] != C:
            raise ValueError(f"Channel mismatch: t has C={C}, ref[1]={ref.shape[1]}")
        b = t.view(1, C, 1, 1, 1)         # (1,C,1,1,1)

    return b.to(device=ref.device, dtype=ref.dtype)