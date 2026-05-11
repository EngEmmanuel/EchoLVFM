# EchoLVFM tests

Unit tests for the core modules. Designed to run **on CPU without any real
data** — they build tiny tensors and a stub UNet in-process so they're safe
to run on any machine (including this 12 GB box).

## Layout

| File | Covers |
|------|--------|
| `test_flows.py`   | `LinearFlow` + `RMMFlow` forward/backward + sampling — uses a `_TinyModel` stub, no UNet3D |
| `test_losses.py`  | `MaskedMSELoss` + `RMMFLoss` — masking semantics, gradient flow, reduction modes |
| `test_model.py`   | `UNet3D` construction, forward shape, attn processors — uses a tiny unet config |

`TestRMMFlow` deliberately sets `prob_default_flow_obj=1.0` so it never hits
the JVP flash-attention branch — the suite runs without CUDA. Real JVP
coverage lives in `benchmarks/bench_training.py` (which does require CUDA).

## Running

```bash
# All tests
pytest tests/

# One file
pytest tests/test_flows.py

# One class
pytest tests/test_flows.py::TestRMMFlow

# One test
pytest tests/test_flows.py::TestRMMFlow::test_forward_returns_dict_with_loss

# Verbose, stop on first failure, show prints
pytest tests/ -vv -x -s

# Filter by keyword (matches test name or parametrize id)
pytest tests/ -k masked
```

## Tests vs benchmarks

This folder is for **correctness** (shapes, masking math, gradient flow).
For **performance** (step time, peak GPU memory, jvp-on vs jvp-off), see
`benchmarks/README.md`. That covers:

- running the whole benchmark suite on the bundled lightweight config
  (`python benchmarks/run.py --lightweight`),
- **running on real CAMUS data** (edit `configs/flow_train/paths/local.yaml`,
  then `python benchmarks/run.py` without `--lightweight`),
- overriding batch size / max frames / rounds / device from the CLI,
- running individual benchmark parametrizations.

## Pytest config

Centralised in `pyproject.toml`:

- `testpaths = ["tests"]` — bare `pytest` runs only this folder.
- `python_files = ["test_*.py", "bench_*.py"]` — lets pytest also collect
  benchmark files under `benchmarks/` when given as an explicit path arg.

If you ever want to switch the default target, change `testpaths`; to add a
new test file pattern, extend `python_files`.

## Adding a new test

1. Drop a `test_*.py` file in this folder.
2. If it needs shared fixtures, use module-level pytest fixtures (not a
   package conftest) — we don't currently have a `tests/conftest.py` and
   don't need one.
3. Keep it CPU-only and data-free: real data + GPU paths belong in
   `benchmarks/`.
