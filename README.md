# EchoLVFM: One-Step Video Generation via Latent Flow Matching for Echocardiogram Synthesis

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.13967-b31b1b.svg)](https://arxiv.org/abs/2603.13967)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

> **🚧 Code Coming Soon**
>
> The paper is currently under review. The code for this project will be released following the review stage. Stay tuned!

---

## Overview

**EchoLVFM** is a one-step video generation framework based on Latent Flow Matching, designed for high-fidelity echocardiogram synthesis. By operating directly in a compressed latent space and learning a continuous flow between noise and the target video distribution, EchoLVFM produces realistic cardiac ultrasound video in a **single forward pass** — dramatically reducing inference time compared to multi-step diffusion-based approaches.

Key highlights:
- ⚡ **One-step Inference** — generate a full video in a single forward pass
- ⏱️ **Variable-length Videos** - supports training on videos of different lengths and allows users to choose the length of the generated video up to a maximum $F$
- 🫀 **Global Parameter Conditioning** — preserves clinically relevant cardiac structures and motion patterns
- 📐 **Latent Video Flow Matching** — efficient training in a low-dimensional latent space


## Visual Demos

### Generation Demo

![Generation demo](docs/media/demos/generation_demo.gif)

Caption: **Generation case** — conditioned on an EF value that is **different** from the real video's EF.

MP4 version (with playback controls/speed): [generation_demo.mp4](docs/media/demos/generation_demo.mp4)

### Reconstruction Demo

![Reconstruction demo](docs/media/demos/reconstruction_demo.gif)

Caption: **Reconstruction case** — conditioned on an EF value that is the **same** as the real video's EF.

MP4 version (with playback controls/speed): [reconstruction_demo.mp4](docs/media/demos/reconstruction_demo.mp4)


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
  author        = {},
  year          = {2026},
  eprint        = {2603.13967},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2603.13967}
}
```

