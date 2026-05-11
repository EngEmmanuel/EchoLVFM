---
library_name: pytorch
tags:
  - video-generation
  - flow-matching
  - echocardiography
  - medical-imaging
pipeline_tag: other
---

# EchoLVFM — Weights

One-step latent video flow matching for echocardiogram synthesis.

- Paper: [arXiv:2603.13967](https://arxiv.org/abs/2603.13967)
- Code: https://github.com/EngEmmanuel/EchoLVFM

This repo holds **weights only**. The training + inference code lives in the
`EchoLVFM` code repository. You need both to run the model.

## Contents

Three independent checkpoints, each in its own subfolder:

| Subfolder | Flow | Inference | Notes |
|-----------|------|-----------|-------|
| `echolvfm_h1/` | RMMFlow | one-step | Adaptive-weighting exponent `h=1` in the training loss |
| `echolvfm_h2/` | RMMFlow | one-step | Adaptive-weighting exponent `h=2` in the training loss |
| `linear/`      | LinearFlow | multi-step ODE | Baseline for comparison |

`h` is a **loss hyperparameter** (the exponent of the adaptive-weighting
term), not a step count. Both RMMFlow variants are one-step generators —
that's the defining property of RMMFlow.

Each subfolder contains:
- `model.safetensors` — the flow-level state dict (~293 MB).
- `config.yaml` — minimal config to rebuild the UNet3D + flow wrapper.

Subfolders load **independently**: a single call only downloads the
requested variant's files (~293 MB), not the whole repo.

## Loading

```python
from utils.hub import load_model_from_hub

flow = load_model_from_hub(
    "EngEmmanuel/EchoLVFM-Weights",
    subfolder="echolvfm_h2",
    device="cuda",
)
```

You also need the paired VAE (`HReynaud/EchoFlow`, subfolder `vae`); see
`vae/util.py::load_vae_and_processor` in the code repo.

## Training data

The underlying models were trained on VAE-encoded latents of the public
[CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) dataset. Please
respect the CAMUS dataset's license and citation requirements when using
these weights.

## Citation

If you use EchoLVFM, please cite the paper:

```bibtex
@article{echolvfm2026,
  title   = {EchoLVFM: One-Step Video Generation via Latent Flow Matching for Echocardiogram Synthesis},
  author  = {Oladokun, Emmanuel and others},
  journal = {arXiv preprint arXiv:2603.13967},
  year    = {2026}
}
```
