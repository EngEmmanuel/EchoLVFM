# jvp_flash_attn_processor.py
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from jvp_flash_attention.jvp_attention import JVPAttn

from diffusers.models.attention_processor import (
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnProcessor,
    Attention
)

class JVPFlashAttnProcessor:
    """
    Processor that:
      - Uses JVPAttn.fwd_dual(query, key, value) for self-attention (forward-mode AD with FlashAttention).
      - Uses scaled_dot_product_attention for cross-attention (AttnProcessor2_0-like), with no mask or dropout.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JVPFlashAttnProcessor requires PyTorch 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # ignored
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # spatial norm
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # (N, C, H, W) -> (N, seq, dim)
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]
            channel = height = width = None  # placeholders

        # encoder states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # head dims
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # (batch, heads, seq, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # optional q/k norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if not attn.is_cross_attention and query.shape[2] >= 32:
            # Self-attention: JVP-capable FlashAttention
            #print(f'query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape}', flush=True)
            #print('using jvp flash attn', flush=True)
            attn_out = JVPAttn.fwd_dual(query, key, value)
        else:
            # Cross-attention: scaled dot-product attention with no mask or dropout
            #print('using sdpa', flush=True)
            attn_out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

        # restore to (batch, seq, heads*head_dim)
        hidden_states = attn_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj (no dropout)
        hidden_states = attn.to_out[0](hidden_states)

        # reshape back to image if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # final scaling
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states