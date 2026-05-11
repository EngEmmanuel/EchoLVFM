"""
Run the EchoLVFM benchmark suite and produce a summary + comparison plots.

Examples:

    # Lightweight smoke (laptop / 12 GB GPU)
    python benchmarks/run.py --lightweight

    # Full-size, real CAMUS data — uses paths=local from configs/flow_train
    python benchmarks/run.py

    # Pass-through extra args go to pytest:
    python benchmarks/run.py --lightweight -- -k jvp -s

Outputs land in `benchmarks/results/<timestamp>/`:
  benchmark.json   raw pytest-benchmark JSON
  summary.md       per-benchmark table (time + peak GPU mem)
  time.png         comparison bar chart of mean step time
  peak_mem.png     comparison bar chart of peak GPU memory
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
RESULTS_ROOT = BENCH_DIR / "results"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lightweight", action="store_true",
                   help="Tiny synthetic batch + tiny UNet3D (laptop-friendly).")
    p.add_argument("--device", default=None, help="Torch device override.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--results-dir", default=None,
                   help="Directory to write outputs into. Default: "
                        "benchmarks/results/<timestamp>/")
    p.add_argument("--keep", action="store_true",
                   help="Keep results dir on pytest failure (default: kept).")
    p.add_argument("pytest_args", nargs=argparse.REMAINDER,
                   help="Extra args passed through to pytest after `--`.")
    return p.parse_args()


def _make_results_dir(override: str | None) -> Path:
    if override:
        out = Path(override).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = RESULTS_ROOT / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _run_pytest(args, results_dir: Path) -> int:
    json_path = results_dir / "benchmark.json"
    cmd = [
        sys.executable, "-m", "pytest", str(BENCH_DIR),
        f"--benchmark-json={json_path}",
        "--benchmark-columns=mean,median,stddev,min,max,rounds",
        "--benchmark-sort=mean",
    ]
    if args.lightweight:
        cmd.append("--lightweight")
    if args.device:
        cmd += ["--device", args.device]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.max_frames is not None:
        cmd += ["--max-frames", str(args.max_frames)]
    if args.rounds is not None:
        cmd += ["--rounds", str(args.rounds)]
    if args.warmup is not None:
        cmd += ["--warmup", str(args.warmup)]
    extra = list(args.pytest_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


_STATS_DROP = {"data", "ops", "iqr_outliers", "stddev_outliers", "outliers",
               "ld15iqr", "hd15iqr", "total", "iterations"}

# Fields in `extra_info` that are shared across every parametrization. They're
# lifted to a single top-level `config` block so the JSON doesn't repeat them
# per-benchmark. Anything NOT in this set stays with the benchmark itself.
_COMMON_EXTRA_KEYS = {"lightweight", "device", "batch_size", "max_frames",
                      "latent_channels", "spatial", "cross_attention_dim",
                      "rounds_configured", "warmup_configured", "unet_kwargs",
                      "axis"}


def _slim_json(json_path: Path) -> None:
    """Shrink pytest-benchmark's raw JSON in-place.

    - Removes `machine_info` (CPUID/flags bloat), `commit_info`, per-benchmark
      `options`, per-round `stats.data`, and outlier fields.
    - Deduplicates the shared regime config out of each benchmark's
      `extra_info` into a single top-level `config` block.
    """
    if not json_path.exists():
        return
    data = json.loads(json_path.read_text())
    data.pop("machine_info", None)
    data.pop("commit_info", None)

    common: dict = {}
    for b in data.get("benchmarks", []):
        b.pop("options", None)
        stats = b.get("stats", {}) or {}
        for k in _STATS_DROP:
            stats.pop(k, None)
        extra = b.get("extra_info", {}) or {}
        for k in list(extra.keys()):
            if k in _COMMON_EXTRA_KEYS:
                # First benchmark wins; subsequent ones should match.
                common.setdefault(k, extra[k])
                del extra[k]
    if common:
        data["config"] = common
    json_path.write_text(json.dumps(data, indent=2))


def _short_label(name: str) -> str:
    """`test_meanflow_training_step[jvp-on]` → `jvp-on`."""
    m = re.search(r"\[([^\]]+)\]", name or "")
    return m.group(1) if m else (name or "")


def _load_records(json_path: Path) -> list[dict]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text())
    common = data.get("config", {}) or {}
    out = []
    for b in data.get("benchmarks", []):
        stats = b.get("stats", {}) or {}
        extra = b.get("extra_info", {}) or {}
        merged = {**common, **extra}
        name = b.get("name") or b.get("fullname")
        out.append({
            "name": name,
            "label": _short_label(name),
            "group": b.get("group"),
            "mean": stats.get("mean"),
            "median": stats.get("median"),
            "stddev": stats.get("stddev"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "rounds": stats.get("rounds"),
            "peak_mib": merged.get("peak_gpu_mem_mib"),
            "device": merged.get("device"),
            "extra": merged,
        })
    return out


def _write_config(records: list[dict], args: argparse.Namespace, out_path: Path) -> None:
    """Dump the benchmark's model/training config as JSON so the results
    folder is self-documenting.
    """
    if not records:
        out_path.write_text("{}\n")
        return
    extra = records[0].get("extra", {}) or {}
    config = {
        "cli_args": {k: v for k, v in vars(args).items() if v is not None
                     and k != "pytest_args"},
        "pytest_pass_through": list(args.pytest_args or []),
        "lightweight": extra.get("lightweight"),
        "device": extra.get("device"),
        "batch_size": extra.get("batch_size"),
        "max_frames": extra.get("max_frames"),
        "latent_channels": extra.get("latent_channels"),
        "spatial": extra.get("spatial"),
        "cross_attention_dim": extra.get("cross_attention_dim"),
        "rounds": extra.get("rounds_configured"),
        "warmup": extra.get("warmup_configured"),
        "unet_kwargs": extra.get("unet_kwargs"),
        "benchmarks": [r["label"] for r in records],
    }
    out_path.write_text(json.dumps(config, indent=2))


def _write_markdown(records: list[dict], out_path: Path) -> None:
    lines = ["# EchoLVFM benchmark summary", ""]
    if not records:
        lines.append("_No benchmarks were collected._")
        out_path.write_text("\n".join(lines))
        return

    devices = sorted({r["device"] for r in records if r.get("device")})
    if devices:
        lines.append(f"**Device(s):** {', '.join(devices)}")
        lines.append("")

    lines.append("| Benchmark | Mean (s) | Median (s) | Stddev (s) | Min (s) | Rounds | Peak GPU mem (MiB) |")
    lines.append("|-----------|---------:|-----------:|-----------:|--------:|-------:|-------------------:|")
    for r in records:
        peak = f"{r['peak_mib']:.1f}" if r["peak_mib"] is not None else "—"
        lines.append(
            f"| `{r['label']}` "
            f"| {r['mean']:.4f} | {r['median']:.4f} | {r['stddev']:.4f} "
            f"| {r['min']:.4f} | {r['rounds']} | {peak} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines))


def _bar_plot(records: list[dict], key: str, ylabel: str, title: str,
              out_path: Path, fmt_value=lambda v: f"{v:.3g}",
              error_key: str | None = None) -> None:
    pairs = [(r["label"], r.get(key)) for r in records if r.get(key) is not None]
    if not pairs:
        return
    labels = [n for n, _ in pairs]
    values = [v for _, v in pairs]
    errors = (
        [r.get(error_key) or 0.0 for r in records if r.get(key) is not None]
        if error_key else None
    )

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels) + 1.5), 4.5))
    bars = ax.bar(range(len(labels)), values, yerr=errors, capsize=4,
                  color=["#1f77b4" if "jvp-on" in lbl else "#ff7f0e"
                         for lbl in labels])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt_value(v),
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summarise(json_path: Path, results_dir: Path, args: argparse.Namespace) -> None:
    _slim_json(json_path)
    records = _load_records(json_path)
    _write_config(records, args, results_dir / "config.json")
    _write_markdown(records, results_dir / "summary.md")
    _bar_plot(records, key="mean",
              ylabel="Mean step time (s)",
              title="Training step time — lower is better",
              out_path=results_dir / "time.png",
              fmt_value=lambda v: f"{v*1000:.1f} ms",
              error_key="stddev")
    _bar_plot(records, key="peak_mib",
              ylabel="Peak GPU memory (MiB)",
              title="Peak GPU memory per step — lower is better",
              out_path=results_dir / "peak_mem.png",
              fmt_value=lambda v: f"{v:.0f} MiB")


def main() -> int:
    args = _parse_args()
    results_dir = _make_results_dir(args.results_dir)
    print(f"\nResults dir: {results_dir}")

    rc = _run_pytest(args, results_dir)
    json_path = results_dir / "benchmark.json"
    _summarise(json_path, results_dir, args)

    print(f"\nConfig:     {results_dir / 'config.json'}")
    print(f"Summary:    {results_dir / 'summary.md'}")
    print(f"Time plot:  {results_dir / 'time.png'}")
    print(f"Mem plot:   {results_dir / 'peak_mem.png'}")

    if rc != 0 and not args.keep and not json_path.exists():
        # Pytest fully failed before producing JSON — cleaner to remove the empty dir.
        shutil.rmtree(results_dir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
