"""Tests for LinearFlow and RMMFlow training and sampling."""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal mock model that matches the UNet3D interface
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """A tiny linear model that mimics the UNet3D call signature.

    Accepts (x, timestep, encoder_hidden_states, cond_image, cond_t=None, mask=None)
    and returns a tensor of shape (B, C, T, H, W) matching x.
    """
    def __init__(self, C: int):
        super().__init__()
        in_ch = 2 * C + 1  # concatenation of x (C) and cond_image (C+1)
        self.conv = nn.Conv3d(in_ch, C, kernel_size=1)

    def forward(self, x, timestep, encoder_hidden_states, cond_image, cond_t=None, mask=None):
        inp = torch.cat([x, cond_image], dim=1)  # (B, 2C+1, T, H, W)
        return self.conv(inp)                      # (B, C, T, H, W)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

C, T, H, W = 1, 4, 8, 8
B = 2
CROSS_ATTN_DIM = 16


@pytest.fixture
def tiny_model():
    return _TinyModel(C=C)


@pytest.fixture
def video_inputs():
    """Returns (x, encoder_hidden_states, cond_image, loss_mask) on CPU."""
    x = torch.randn(B, C, T, H, W)
    encoder_hidden_states = torch.randn(B, 1, CROSS_ATTN_DIM)
    cond_image = torch.randn(B, C + 1, T, H, W)  # C latent + 1 mask channel
    loss_mask = torch.ones(B, T)
    loss_mask[0, -1] = 0.0  # pad last frame for first sample
    return x, encoder_hidden_states, cond_image, loss_mask


# ---------------------------------------------------------------------------
# LinearFlow tests
# ---------------------------------------------------------------------------

class TestLinearFlow:

    def test_forward_returns_scalar_loss(self, tiny_model, video_inputs):
        from src.flows.linear_flow import LinearFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = LinearFlow(tiny_model)
        loss = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        assert loss.ndim == 0, "LinearFlow.forward should return a scalar loss"
        assert loss.item() >= 0.0

    def test_forward_without_loss_mask(self, tiny_model, video_inputs):
        from src.flows.linear_flow import LinearFlow
        x, ehs, cond_image, _ = video_inputs
        flow = LinearFlow(tiny_model)
        loss = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=None)
        assert loss.ndim == 0

    def test_forward_gradients_flow(self, tiny_model, video_inputs):
        from src.flows.linear_flow import LinearFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = LinearFlow(tiny_model)
        loss = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        loss.backward()
        grad_norm = sum(
            p.grad.norm().item()
            for p in tiny_model.parameters()
            if p.grad is not None
        )
        assert grad_norm > 0.0, "Gradients should flow through LinearFlow"

    def test_forward_stores_data_shape(self, tiny_model, video_inputs):
        from src.flows.linear_flow import LinearFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = LinearFlow(tiny_model)
        _ = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        assert flow.data_shape is not None
        assert list(flow.data_shape) == [C, T, H, W]

    def test_sample_returns_correct_shape(self, tiny_model, video_inputs):
        pytest.importorskip("torchdiffeq")
        from src.flows.linear_flow import LinearFlow
        x, ehs, cond_image, _ = video_inputs
        flow = LinearFlow(tiny_model, data_shape=(C, T, H, W))
        with torch.no_grad():
            sample = flow.sample(
                encoder_hidden_states=ehs,
                batch_size=B,
                data_shape=(C, T, H, W),
                cond_image=cond_image,
                steps=2,
                odeint_kwargs=dict(atol=1e-3, rtol=1e-3, method='euler'),
            )
        assert sample.shape == (B, C, T, H, W), (
            f"Expected shape {(B, C, T, H, W)}, got {sample.shape}"
        )


# ---------------------------------------------------------------------------
# RMMFlow tests
# ---------------------------------------------------------------------------

class TestRMMFlow:

    def test_forward_returns_dict_with_loss(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, loss_mask = video_inputs
        # prob_default_flow_obj=1.0 avoids JVP path for a fast CPU test
        flow = RMMFlow(tiny_model, prob_default_flow_obj=1.0)
        out = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        assert isinstance(out, dict), "RMMFlow.forward should return a dict"
        assert 'loss' in out, "Output dict must contain 'loss'"
        assert 'flow_loss' in out

    def test_forward_loss_is_scalar(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = RMMFlow(tiny_model, prob_default_flow_obj=1.0)
        out = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        assert out['loss'].ndim == 0
        assert out['loss'].item() >= 0.0

    def test_forward_without_mask(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, _ = video_inputs
        flow = RMMFlow(tiny_model, prob_default_flow_obj=1.0)
        out = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=None)
        assert out['loss'].item() >= 0.0

    def test_forward_gradients_flow(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = RMMFlow(tiny_model, prob_default_flow_obj=1.0)
        out = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        out['loss'].backward()
        grad_norm = sum(
            p.grad.norm().item()
            for p in tiny_model.parameters()
            if p.grad is not None
        )
        assert grad_norm > 0.0, "Gradients should flow through RMMFlow"

    def test_forward_stores_data_shape(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = RMMFlow(tiny_model, prob_default_flow_obj=1.0)
        _ = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        assert flow.data_shape is not None
        assert list(flow.data_shape) == [C, T, H, W]

    def test_sample_returns_correct_shape(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, _ = video_inputs
        flow = RMMFlow(tiny_model, data_shape=(C, T, H, W), prob_default_flow_obj=1.0)
        sample = flow.sample(
            encoder_hidden_states=ehs,
            cond_image=cond_image,
            batch_size=B,
            data_shape=(C, T, H, W),
            steps=1,
        )
        assert sample.shape == (B, C, T, H, W), (
            f"Expected shape {(B, C, T, H, W)}, got {sample.shape}"
        )

    def test_slow_sample_returns_correct_shape(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, _ = video_inputs
        flow = RMMFlow(tiny_model, data_shape=(C, T, H, W), prob_default_flow_obj=1.0)
        with torch.no_grad():
            sample = flow.sample(
                encoder_hidden_states=ehs,
                cond_image=cond_image,
                batch_size=B,
                data_shape=(C, T, H, W),
                steps=2,
            )
        assert sample.shape == (B, C, T, H, W)

    def test_cfg_forward_runs(self, tiny_model, video_inputs):
        """Test that CFG training path runs without error."""
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = RMMFlow(
            tiny_model,
            prob_default_flow_obj=1.0,
            uncond_prob=0.1,
        )
        out = flow(x, encoder_hidden_states=ehs, cond_image=cond_image, loss_mask=loss_mask)
        assert out['loss'].item() >= 0.0

    def test_logit_normal_time_sampler(self, tiny_model, video_inputs):
        from src.flows.rmm_flow import RMMFlow
        x, ehs, cond_image, loss_mask = video_inputs
        flow = RMMFlow(
            tiny_model,
            prob_default_flow_obj=1.0,
            use_logit_normal_sampler=True,
        )
        times = flow.sample_times(B)
        assert times.shape == (B,)
        assert (times >= 0.).all() and (times <= 1.).all(), "Times must be in [0, 1]"
