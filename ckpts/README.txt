Model Checkpoints
=================

Pretrained model checkpoints are not stored in this repository due to their size.

Hugging Face Hub (recommended)
------------------------------
Weights are published at:

    https://huggingface.co/EngEmmanuel/EchoLVFM-Weights

The repo contains three subfolders (echolvfm_h1, echolvfm_h2, linear). You
can load any variant directly from Python without manually downloading
files:

    from utils.hub import load_model_from_hub

    flow = load_model_from_hub(
        "EngEmmanuel/EchoLVFM-Weights",
        subfolder="echolvfm_h2",
        device="cuda",
    )

This is the easiest path for inference. Only the requested subfolder is
downloaded (~293 MB).

GitHub Releases (alternative)
-----------------------------
Lightning .ckpt files are also attached to the GitHub release assets, for
users who want to resume training or inspect the raw Lightning state:

1. Go to the Releases page of this repository on GitHub.
2. Download the checkpoint file(s) from the latest release.
3. Place them in this directory (ckpts/).

Usage
-----
Once a Lightning .ckpt is downloaded, point the evaluator at the training run directory:

    from evaluation.functions import load_model_from_run
    model, ckpt_path = load_model_from_run(run_dir="outputs/<run>", dummy_data=sample)

Or use the inference benchmark:

    bash scripts/run_benchmark.sh --run_dir outputs/<run> --ckpt_name <name>.ckpt

Checkpoint Format
-----------------
Checkpoints are saved by PyTorch Lightning and contain:
  - state_dict: model weights (flow wrapper + UNet3D)
  - epoch, global_step: training progress

The flow type (RMMFlow or LinearFlow) and all model hyperparameters
are stored in the accompanying Hydra config at:
    outputs/<run>/.hydra/config.yaml
