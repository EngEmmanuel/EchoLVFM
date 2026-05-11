"""Tests for UNet3D model architecture.

These tests require diffusers and torch. They use a small model config to
keep runtime short (CPU only).
"""

import pytest
import torch

diffusers = pytest.importorskip("diffusers", reason="diffusers not installed")


# ---------------------------------------------------------------------------
# Minimal UNet3D config for testing
# ---------------------------------------------------------------------------

TINY_UNET_CFG = dict(
    sample_size=8,
    in_channels=3,    # 2*C+1 where C=1 (one latent channel for testing)
    out_channels=1,
    down_block_types=(
        "CrossAttnDownBlockSpatioTemporal",
        "DownBlockSpatioTemporal",
    ),
    up_block_types=(
        "UpBlockSpatioTemporal",
        "CrossAttnUpBlockSpatioTemporal",
    ),
    block_out_channels=(32, 64),
    layers_per_block=1,
    cross_attention_dim=32,
    transformer_layers_per_block=1,
    num_attention_heads=(2, 4),
    num_frames=4,
)

# Test tensor dimensions
B, C, T, H, W = 1, 1, 4, 8, 8
CROSS_ATTN_DIM = 32


@pytest.fixture(scope="module")
def tiny_unet():
    """Instantiate a small UNet3D once for the test module."""
    from src.model import UNet3D
    model = UNet3D(**TINY_UNET_CFG)
    model.eval()
    return model


@pytest.fixture
def model_inputs():
    """Returns a dict of inputs matching the UNet3D forward signature."""
    return dict(
        x=torch.randn(B, C, T, H, W),
        timestep=torch.rand(B),
        encoder_hidden_states=torch.randn(B, 1, CROSS_ATTN_DIM),
        cond_image=torch.randn(B, C + 1, T, H, W),  # C latent + 1 mask channel
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUNet3DConstruction:
    def test_instantiation(self):
        from src.model import UNet3D
        model = UNet3D(**TINY_UNET_CFG)
        assert model is not None

    def test_parameter_count_is_positive(self, tiny_unet):
        n_params = sum(p.numel() for p in tiny_unet.parameters())
        assert n_params > 0

    def test_has_expected_sub_modules(self, tiny_unet):
        assert hasattr(tiny_unet, "conv_in")
        assert hasattr(tiny_unet, "conv_out")
        assert hasattr(tiny_unet, "down_blocks")
        assert hasattr(tiny_unet, "up_blocks")
        assert hasattr(tiny_unet, "mid_block")
        assert hasattr(tiny_unet, "time_embedding")


class TestUNet3DForward:
    def test_output_shape(self, tiny_unet, model_inputs):
        with torch.no_grad():
            out = tiny_unet(**model_inputs)
        assert out.shape == (B, C, T, H, W), (
            f"Expected output shape {(B, C, T, H, W)}, got {out.shape}"
        )

    def test_float_timestep(self, tiny_unet, model_inputs):
        """UNet3D should handle scalar float timestep."""
        inputs = dict(model_inputs)
        inputs['timestep'] = 0.5
        with torch.no_grad():
            out = tiny_unet(**inputs)
        assert out.shape == (B, C, T, H, W)

    def test_with_cond_t(self, tiny_unet, model_inputs):
        """RMMFlow passes cond_t (delta time) to the model."""
        with torch.no_grad():
            out = tiny_unet(**model_inputs, cond_t=torch.rand(B))
        assert out.shape == (B, C, T, H, W)

    def test_with_scalar_cond_t(self, tiny_unet, model_inputs):
        with torch.no_grad():
            out = tiny_unet(**model_inputs, cond_t=1.0)
        assert out.shape == (B, C, T, H, W)

    def test_output_is_finite(self, tiny_unet, model_inputs):
        with torch.no_grad():
            out = tiny_unet(**model_inputs)
        assert torch.isfinite(out).all(), "Model output should be finite"

    def test_gradients_flow(self, tiny_unet, model_inputs):
        tiny_unet.train()
        out = tiny_unet(**model_inputs)
        loss = out.mean()
        loss.backward()
        grad_norm = sum(
            p.grad.norm().item()
            for p in tiny_unet.parameters()
            if p.grad is not None
        )
        tiny_unet.eval()
        assert grad_norm > 0.0, "Gradients should flow through UNet3D"


class TestUNet3DAttnProcessors:
    def test_attn_processors_property(self, tiny_unet):
        procs = tiny_unet.attn_processors
        assert isinstance(procs, dict)
        assert len(procs) > 0

    def test_set_attn_processor(self, tiny_unet):
        from diffusers.models.attention_processor import AttnProcessor
        tiny_unet.set_attn_processor(AttnProcessor())
        procs = tiny_unet.attn_processors
        for proc in procs.values():
            assert isinstance(proc, AttnProcessor)
