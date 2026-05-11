import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes masked MSE loss. If mask is None, falls back to standard MSE.
        Args:
            input (Tensor): Predicted tensor.
            target (Tensor): Ground truth tensor (same shape as input).
            mask:   (B, T) binary {0,1} where 1 means "include in loss"

        Returns:
            Tensor: Scalar loss if reduction != "none", else elementwise loss.
        """
        if mask is None:
            # fallback: behave like standard MSELoss
            return F.mse_loss(input, target, reduction=self.reduction)

        # Expand mask to broadcast over C,H,W
        mask = mask[:, None, :, None, None]  # (B,1,T,1,1)
        B, C, T, H, W = input.shape

        diff = (input - target) ** 2
        masked_diff = diff * mask  # masked out entries = 0

        if self.reduction == "mean":
            return masked_diff.sum() / (mask.sum() * C * H * W).clamp(min=1)
        elif self.reduction == "sum":
            return masked_diff.sum()
        elif self.reduction == "none":
            return masked_diff


## RMM FLOW LOSS

class RMMFLoss(nn.Module):
    """
    Mask-aware loss for video flow matching.

    Args:
        use_adaptive_loss_weight (bool): apply per-sample weight w = 1/(mse+eps)^p.
        adaptive_loss_weight_p (float): p in the adaptive weight.
        eps (float): small constant for numerical stability.
        add_recon_loss (bool): add masked reconstruction term.
        recon_loss_weight (float): weight for the reconstruction term.
    """
    def __init__(
        self,
        use_adaptive_loss_weight: bool = True,
        adaptive_loss_weight_p: float = 0.5,
        eps: float = 1e-3,
        add_recon_loss: bool = False,
        recon_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.use_adaptive_loss_weight = use_adaptive_loss_weight
        self.adaptive_loss_weight_p = adaptive_loss_weight_p
        self.eps = eps
        self.add_recon_loss = add_recon_loss
        self.recon_loss_weight = recon_loss_weight

    @staticmethod
    def _masked_per_sample_mse(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        Per-sample masked MSE averaged over C,T,H,W for each sample.
        pred/target: (B, C, T, H, W)
        loss_mask:  (B, T) with 1 for valid frames, 0 for pads
        Returns: (B,) tensor of per-sample masked means.
        """
        B, C, T, H, W = pred.shape
        if loss_mask is None:
            m = torch.ones((B, T), dtype=pred.dtype, device=pred.device)
        else:
            m = loss_mask.to(dtype=pred.dtype, device=pred.device)  # (B,T)
        m_exp = m[:, None, :, None, None]                        # (B,1,T,1,1)

        diff2 = (pred - target).pow(2) * m_exp                   # mask out pads
        num = diff2.sum(dim=(1, 2, 3, 4))                        # (B,)
        denom = (m.sum(dim=1) * C * H * W).clamp_min(1)          # (B,)
        return num / denom

    def forward(
        self,
        pred: torch.Tensor,         # u_pred;  (B, C, T, H, W)
        flow: torch.Tensor,         # v(_hat); (B, C, T, H, W)
        integral: torch.Tensor,     # (t-r)*[J.(v, 0 , 1)] (B, C, T, H, W)
        loss_mask: torch.Tensor,    # (B, T) -> 1 = include, 0 = pad
        *,
        noised_data: torch.Tensor | None = None,   # for recon term
        data: torch.Tensor | None = None,          # GT for recon
        padded_times: torch.Tensor | None = None,  # broadcastable to (B,C,T,H,W)
    ):
        """
        Returns:
            total_loss (scalar) or (total_loss, (flow_loss, recon_loss))
        """
        # ----- flow loss (masked) -----
        target = flow - integral # u_tgt
        per_sample_mse = self._masked_per_sample_mse(pred, target, loss_mask)  # (B,)

        if self.use_adaptive_loss_weight:
            # Effect (adaptive weight w = 1/(||Δ||^2 + eps)^p, using sg(w)·||Δ||^2):
            #   p = 0    → plain L2 (no reweighting; outliers influence more).
            #   p ≈ 0.5  → pseudo-Huber–like robustness (down-weights large errors).
            #   p ≈ 1.0  → strong robustness; emphasizes small residuals/fine detail.

            p = self.adaptive_loss_weight_p
            w = 1.0 / (per_sample_mse + self.eps).pow(p)       # (B,)
            flow_loss = (per_sample_mse * w.detach()).mean()
        else:
            flow_loss = per_sample_mse.mean()

        # ----- optional masked reconstruction loss -----
        recon_loss = pred.new_zeros(())
        if self.add_recon_loss:
            if (noised_data is None) or (data is None) or (padded_times is None):
                raise ValueError("noised_data, data, and padded_times must be provided when add_recon_loss=True.")

            pred_data = noised_data - (pred + integral) * padded_times
            recon_per_sample = self._masked_per_sample_mse(pred_data, data, loss_mask)  # (B,)
            recon_loss = recon_per_sample.mean()

        total = flow_loss + self.recon_loss_weight * recon_loss

        out = {'loss': total, 'flow_loss': flow_loss}
        if self.add_recon_loss:
            out['recon_loss'] = recon_loss
        return out
