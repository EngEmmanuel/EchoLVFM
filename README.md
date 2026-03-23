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


---

## Paper

> **EchoLVFM: One-Step Video Generation via Latent Flow Matching for Echocardiogram Synthesis**
>
> Available on arXiv: https://arxiv.org/abs/2603.13967

## Videos

Sample and demo videos are stored in the [`videos/`](videos/) folder.

### Adding Videos Without Cloning

You can upload videos directly through GitHub's web interface — no Git knowledge required:

1. Go to the [`videos/`](https://github.com/EngEmmanuel/EchoLVFM/tree/main/videos) folder on GitHub.
2. Click **"Add file"** → **"Upload files"**.
3. Drag and drop your video files (or click to browse).
4. Add a short commit message and click **"Commit changes"**.

> **Note:** GitHub supports files up to **100 MB** via the web interface.  
> This repository uses [Git LFS](https://git-lfs.github.com/) for larger video files (up to 2 GB), which requires a local Git + LFS setup.  
> See [`videos/README.md`](videos/README.md) for more details.

---

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

