import re
import pandas as pd

from typing import Callable, Optional, Tuple


def extract_epoch_step_from_checkpoint_str(s: str) -> Tuple[float, float]:
    """Return (epoch, step) parsed from a checkpoint name string; NaNs if absent."""
    if pd.isna(s):
        return float('nan'), float('nan')
    me = re.search(r'epoch=([0-9]+(?:\.[0-9]+)?)', s)
    ms = re.search(r'step=([0-9]+(?:\.[0-9]+)?)', s)
    return (float(me.group(1)) if me else float('nan'),
            float(ms.group(1)) if ms else float('nan'))


def resolve_last_checkpoint_positions(
    df: pd.DataFrame,
    load_last_ckpt_fn: Optional[Callable[[], dict]] = None,
) -> pd.DataFrame:
    """For rows where checkpoint == 'last', set epoch/step from the checkpoint payload.

    Falls back to placing 'last' after the current maximum numeric epoch/step if
    the loader is unavailable or the checkpoint lacks the expected keys.
    """
    last_mask = df['checkpoint'].astype(str).str.contains(r'\blast\b', na=False)
    if not last_mask.any():
        return df

    max_epoch = df['epoch'].dropna().max()
    max_step  = df['step'].dropna().max()
    max_epoch = 0.0 if pd.isna(max_epoch) else max_epoch
    max_step  = 0.0 if pd.isna(max_step)  else max_step

    try:
        ckpt = load_last_ckpt_fn()
        ckpt_epoch = ckpt.get('epoch', ckpt.get('epoch_idx', None))
        ckpt_step  = ckpt.get('global_step', ckpt.get('step', None))
        if ckpt_epoch is None or ckpt_step is None:
            raise KeyError("checkpoint missing epoch/global_step")
        df.loc[last_mask, 'epoch'] = float(ckpt_epoch)
        df.loc[last_mask, 'step']  = float(ckpt_step)
    except Exception:
        df.loc[last_mask, 'epoch'] = max_epoch + 1.0
        df.loc[last_mask, 'step']  = max_step  + 1.0

    return df
