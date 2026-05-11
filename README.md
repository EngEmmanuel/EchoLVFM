# EchoLVFM: One-Step Video Generation via Latent Flow Matching for Echocardiogram Synthesis

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.13967-b31b1b.svg)](https://arxiv.org/abs/2603.13967)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📰 News

- 🎉 **May 2026** EchoLVFM has been **early-accepted at MICCAI 2026**, selected from the top 9% of 4,601 submissions!
- 🫀 **May 2026** Try the live demo on [Hugging Face Spaces](https://huggingface.co/spaces/EngEmmanuel/EchoLVFM).

---

## Overview

**EchoLVFM** is a one-step video generation framework based on Latent Flow Matching, designed for high-fidelity echocardiogram synthesis. By operating directly in a compressed latent space and learning a continuous flow between noise and the target video distribution, EchoLVFM produces realistic cardiac ultrasound video in a **single forward pass** — dramatically reducing inference time compared to multi-step diffusion-based approaches.

Key highlights:
- ⚡ **One-step Inference** — generate a full video in a single forward pass
- ⏱️ **Variable-length Videos** - supports training on videos of different lengths and allows users to choose the length of the generated video up to a maximum $F$
- 🫀 **Global Parameter Conditioning** — preserves clinically relevant cardiac structures and motion patterns
- 📐 **Latent Video Flow Matching** — efficient training in a low-dimensional latent space


> 🎮 **Try it out!** Run EchoLVFM live in your browser on the [🫀 Hugging Face Space](https://huggingface.co/spaces/EngEmmanuel/EchoLVFM); no setup, no GPU required.

## Visual Demos

### Generation Demo

⏳ *GIF loading… please wait a moment*

![Generation demo](docs/media/demos/generation_demo.gif)

Caption: **Generation case** — conditioned on an EF value that is **different** from the real video's EF.

MP4 version (with playback controls/speed): [generation_demo.mp4](docs/media/demos/generation_demo.mp4)

### Reconstruction Demo

⏳ *GIF loading… please wait a moment*

![Reconstruction demo](docs/media/demos/reconstruction_demo.gif)

Caption: **Reconstruction case** — conditioned on an EF value that is the **same** as the real video's EF.

MP4 version (with playback controls/speed): [reconstruction_demo.mp4](docs/media/demos/reconstruction_demo.mp4)

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Windows — Triton workaround

`jvp-flash-attention` transitively imports `triton`, and upstream Triton does
not publish Windows wheels. Install the community Windows fork instead — it
exposes the same `import triton` module, so `jvp-flash-attention` Just Works.

Pick the Triton version that matches your PyTorch (see the
[triton-windows compatibility table](https://github.com/triton-lang/triton-windows#readme)):

| PyTorch | triton-windows |
|---------|----------------|
| 2.7     | `<3.4`         |
| 2.8     | `<3.5`         |
| 2.9     | `<3.6`         |

```bash
pip uninstall triton                      # if a stub was previously installed
pip install -U "triton-windows<3.5"       # example: for torch 2.8
```

Requirements: NVIDIA GPU ≥ sm_75 (RTX 20xx+) with CUDA 12, Visual C++
Redistributable (`vc_redist.x64.exe`), and Windows long-path support enabled.

---

## Data Preparation

The model trains on VAE latent representations of the [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/).
Pre-encode your videos with the pretrained cardiac VAE:

```python
from vae.util import load_vae_and_processor
vae, processor = load_vae_and_processor("HReynaud/EchoFlow", subfolder="vae", device="cuda")
```

Each video should be saved as a `.pt` file containing `{'mu': ..., 'std': ...}` tensors
of shape `(T, C, H, W)`. You also need a `metadata.csv` with columns `video_name`, `split`
(train/val), and `EF_AL` (ejection fraction as a percentage).

A handful of pre-encoded CAMUS samples ship in `sample_data/CAMUS_Latents_4f4/`
so you can try the code without re-encoding anything yourself.

---

## Pretrained Weights

Weights are published on the Hugging Face Hub at
[**huggingface.co/EngEmmanuel/EchoLVFM-Weights**](https://huggingface.co/EngEmmanuel/EchoLVFM-Weights)
and load directly from Python without a manual download:

```python
from utils.hub import load_model_from_hub

flow = load_model_from_hub(
    "EngEmmanuel/EchoLVFM-Weights",
    subfolder="echolvfm_h2",   # also: echolvfm_h1, linear
    device="cuda",
)
```

Each subfolder is independent; only the requested variant's files
(~293 MB) are downloaded. See `ckpts/README.txt` for the alternative
Lightning `.ckpt` download path via GitHub Releases.

---

## Training

1. Edit `configs/flow_train/paths/local.yaml` with your data and output paths.
2. Run:

```bash
python trainer.py paths=local
```

Override any value from the command line, e.g.:

```bash
python trainer.py paths=local flow=linear trainer.lr=1e-4 dataset.batch_size=4
```

---

## Paper

> **EchoLVFM: One-Step Video Generation via Latent Flow Matching for Echocardiogram Synthesis**
>
> Available on arXiv: https://arxiv.org/abs/2603.13967

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{echolvfm2026,
  title         = {EchoLVFM: One-Step Video Generation via Latent Flow Matching for Echocardiogram Synthesis},
  author        = {Oladokun, Emmanuel and Thomas, Sarina and Šprem, Jurica and Grau, Vicente},
  year          = {2026},
  eprint        = {2603.13967},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2603.13967}
}
```

## Acknowledgements
- Wang, P.: [rectified-flow-pytorch](https://gitlab.com/lucidrains/rectified-flow-pytorch). 
- Morehead, A.: [JVP Flash Attention](https://github.com/amorehead/jvp_flash_attention).
