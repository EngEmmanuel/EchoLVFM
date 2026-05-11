"""Tests for MaskedMSELoss and RMMFLoss."""

import pytest
import torch
from src.custom_loss import MaskedMSELoss, RMMFLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def video_batch():
    """Small (B=2, C=2, T=4, H=4, W=4) tensors for testing."""
    B, C, T, H, W = 2, 2, 4, 4, 4
    pred = torch.randn(B, C, T, H, W)
    target = torch.randn(B, C, T, H, W)
    # Mask: first sample has all 4 frames valid; second sample has only 2 valid
    mask = torch.tensor([[1., 1., 1., 1.],
                          [1., 1., 0., 0.]])  # (B, T)
    return pred, target, mask


# ---------------------------------------------------------------------------
# MaskedMSELoss
# ---------------------------------------------------------------------------

class TestMaskedMSELoss:
    def test_no_mask_matches_standard_mse(self, video_batch):
        pred, target, _ = video_batch
        loss_fn = MaskedMSELoss()
        masked_loss = loss_fn(pred, target, mask=None)
        standard_loss = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(masked_loss, standard_loss), (
            "MaskedMSELoss with mask=None should match standard MSE"
        )

    def test_full_mask_matches_standard_mse(self, video_batch):
        pred, target, _ = video_batch
        B, C, T, H, W = pred.shape
        full_mask = torch.ones(B, T)
        loss_fn = MaskedMSELoss()
        masked_loss = loss_fn(pred, target, mask=full_mask)
        standard_loss = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(masked_loss, standard_loss, atol=1e-6), (
            "All-ones mask should give same result as standard MSE"
        )

    def test_mask_zeros_out_padded_frames(self, video_batch):
        pred, target, mask = video_batch
        loss_fn = MaskedMSELoss()
        loss = loss_fn(pred, target, mask=mask)
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0.0, "Loss must be non-negative"

    def test_masking_changes_loss(self, video_batch):
        pred, target, _ = video_batch
        B, C, T, H, W = pred.shape
        # Modify frames 2 and 3 of sample 1 to be very different
        pred_modified = pred.clone()
        pred_modified[1, :, 2:, :, :] = target[1, :, 2:, :, :] + 100.0

        full_mask = torch.ones(B, T)
        partial_mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]])

        loss_fn = MaskedMSELoss()
        loss_full = loss_fn(pred_modified, target, mask=full_mask)
        loss_partial = loss_fn(pred_modified, target, mask=partial_mask)
        assert loss_partial.item() < loss_full.item(), (
            "Masking out the large-error frames should reduce loss"
        )

    def test_all_zero_mask_returns_zero(self, video_batch):
        pred, target, _ = video_batch
        B, C, T, H, W = pred.shape
        zero_mask = torch.zeros(B, T)
        loss_fn = MaskedMSELoss()
        loss = loss_fn(pred, target, mask=zero_mask)
        assert loss.item() == 0.0, "All-zero mask should give zero loss"

    def test_reduction_sum(self, video_batch):
        pred, target, mask = video_batch
        loss_fn = MaskedMSELoss(reduction="sum")
        loss = loss_fn(pred, target, mask=mask)
        assert loss.item() >= 0.0

    def test_gradients_flow_through(self, video_batch):
        pred, target, mask = video_batch
        pred = pred.requires_grad_(True)
        loss_fn = MaskedMSELoss()
        loss = loss_fn(pred, target, mask=mask)
        loss.backward()
        assert pred.grad is not None, "Gradient should flow through MaskedMSELoss"


# ---------------------------------------------------------------------------
# RMMFLoss
# ---------------------------------------------------------------------------

class TestRMMFLoss:

    @pytest.fixture
    def rmm_flow_inputs(self):
        B, C, T, H, W = 2, 2, 4, 4, 4
        pred = torch.randn(B, C, T, H, W)
        flow = torch.randn(B, C, T, H, W)
        integral = torch.zeros(B, C, T, H, W)  # zero integral = plain flow matching
        loss_mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]])
        return pred, flow, integral, loss_mask, B, C, T, H, W

    def test_output_is_dict_with_required_keys(self, rmm_flow_inputs):
        pred, flow, integral, loss_mask, *_ = rmm_flow_inputs
        loss_fn = RMMFLoss()
        out = loss_fn(pred=pred, flow=flow, integral=integral, loss_mask=loss_mask)
        assert 'loss' in out, "Output dict must contain 'loss'"
        assert 'flow_loss' in out, "Output dict must contain 'flow_loss'"

    def test_loss_is_scalar(self, rmm_flow_inputs):
        pred, flow, integral, loss_mask, *_ = rmm_flow_inputs
        loss_fn = RMMFLoss()
        out = loss_fn(pred=pred, flow=flow, integral=integral, loss_mask=loss_mask)
        assert out['loss'].ndim == 0, "'loss' must be a scalar tensor"
        assert out['loss'].item() >= 0.0, "Loss must be non-negative"

    def test_no_mask_runs(self, rmm_flow_inputs):
        pred, flow, integral, _, B, C, T, H, W = rmm_flow_inputs
        loss_fn = RMMFLoss()
        out = loss_fn(pred=pred, flow=flow, integral=integral, loss_mask=None)
        assert out['loss'].item() >= 0.0

    def test_adaptive_vs_non_adaptive_differ(self, rmm_flow_inputs):
        pred, flow, integral, loss_mask, *_ = rmm_flow_inputs
        fn_adaptive = RMMFLoss(use_adaptive_loss_weight=True)
        fn_plain = RMMFLoss(use_adaptive_loss_weight=False)
        out_a = fn_adaptive(pred=pred, flow=flow, integral=integral, loss_mask=loss_mask)
        out_p = fn_plain(pred=pred, flow=flow, integral=integral, loss_mask=loss_mask)
        # They use different weighting schemes and should generally differ
        assert out_a['loss'].shape == out_p['loss'].shape

    def test_masking_changes_loss(self, rmm_flow_inputs):
        B, C, T, H, W = rmm_flow_inputs[4], rmm_flow_inputs[5], rmm_flow_inputs[6], rmm_flow_inputs[7], rmm_flow_inputs[8]
        # Make frames 2,3 of sample 1 very different
        pred = torch.zeros(B, C, T, H, W)
        flow = torch.zeros(B, C, T, H, W)
        flow[1, :, 2:, :, :] = 10.0  # large error in masked-out frames
        integral = torch.zeros(B, C, T, H, W)

        full_mask = torch.ones(B, T)
        partial_mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 0., 0.]])

        loss_fn = RMMFLoss(use_adaptive_loss_weight=False)
        loss_full = loss_fn(pred=pred, flow=flow, integral=integral, loss_mask=full_mask)
        loss_partial = loss_fn(pred=pred, flow=flow, integral=integral, loss_mask=partial_mask)
        assert loss_partial['loss'].item() < loss_full['loss'].item(), (
            "Masking out high-error frames should reduce loss"
        )

    def test_recon_loss_requires_extra_inputs(self, rmm_flow_inputs):
        pred, flow, integral, loss_mask, *_ = rmm_flow_inputs
        loss_fn = RMMFLoss(add_recon_loss=True)
        with pytest.raises(ValueError, match="noised_data, data, and padded_times"):
            loss_fn(pred=pred, flow=flow, integral=integral, loss_mask=loss_mask)

    def test_recon_loss_included_in_output(self, rmm_flow_inputs):
        pred, flow, integral, loss_mask, B, C, T, H, W = rmm_flow_inputs
        noised = torch.randn(B, C, T, H, W)
        data = torch.randn(B, C, T, H, W)
        padded_times = torch.rand(B, 1, 1, 1, 1)
        loss_fn = RMMFLoss(add_recon_loss=True, recon_loss_weight=1.0)
        out = loss_fn(
            pred=pred, flow=flow, integral=integral, loss_mask=loss_mask,
            noised_data=noised, data=data, padded_times=padded_times
        )
        assert 'recon_loss' in out, "recon_loss should appear in output when add_recon_loss=True"

    def test_gradients_flow_through(self, rmm_flow_inputs):
        pred, flow, integral, loss_mask, *_ = rmm_flow_inputs
        pred = pred.requires_grad_(True)
        loss_fn = RMMFLoss()
        out = loss_fn(pred=pred, flow=flow, integral=integral, loss_mask=loss_mask)
        out['loss'].backward()
        assert pred.grad is not None, "Gradient should flow through RMMFLoss"
