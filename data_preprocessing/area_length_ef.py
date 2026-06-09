"""
Area-length ejection fraction from LV segmentation masks.

ED is taken as the largest-LV-area frame and ES as the smallest.
Volumes use the single-plane area-length formula V = 8/(3π)·A²/L,
and the biplane formula V = 8/(3π)·A2·A4/Lavg when both views are available.


Area-length calculation code inspired by: https://github.com/KhaledElrefaey/EchoFlow-Net/blob/main/notebook/EchoFlow-Net.ipynb
"""

import numpy as np


def _principal_axis_length(mask_bin: np.ndarray) -> float:
    """Length (px) of the mask's principal axis via PCA on the foreground pixels."""
    ys, xs = np.nonzero(mask_bin.astype(bool))
    if xs.size < 10:
        return float('nan')
    X = np.stack([xs, ys], axis=1).astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[0]
    return float(proj.max() - proj.min())


def _ed_es_measurements(masks: np.ndarray, lv_label: int):
    """Return (area, length) for the ED and ES frames of a mask sequence (T, H, W).

    ED = frame with the largest LV area, ES = frame with the smallest.
    """
    masks_lv = (masks == lv_label).astype(np.uint8)
    areas = np.array([float(f.sum()) for f in masks_lv])
    lengths = np.array([_principal_axis_length(f) for f in masks_lv])
    ed_idx, es_idx = int(np.argmax(areas)), int(np.argmin(areas))
    return (areas[ed_idx], lengths[ed_idx]), (areas[es_idx], lengths[es_idx])


def _single_plane_volume(area: float, length: float) -> float:
    """Single-plane area-length volume: V = 8/(3π)·A²/L."""
    if np.isfinite(area) and np.isfinite(length) and length > 0:
        return 8.0 / (3.0 * np.pi) * (area * area) / length
    return float('nan')


def _biplane_volume(A2: float, L2: float, A4: float, L4: float) -> float:
    """Area-length biplane volume.

    Uses 8/(3π)·A2·A4/Lavg (Lavg = mean of the two axis lengths) when both views
    are valid, otherwise falls back to the mean of the available single-plane volumes.
    """
    vals = []
    v2 = _single_plane_volume(A2, L2)
    v4 = _single_plane_volume(A4, L4)
    if np.isfinite(v2):
        vals.append(v2)
    if np.isfinite(v4):
        vals.append(v4)
    if all(np.isfinite([A2, L2, A4, L4])) and L2 > 0 and L4 > 0:
        Lavg = 0.5 * (L2 + L4)
        vals.append(8.0 / (3.0 * np.pi) * (A2 * A4) / Lavg)
    if not vals:
        return float('nan')
    return float(vals[-1] if len(vals) == 3 else np.mean(vals))


def _ef_percent(ed_vol: float, es_vol: float) -> float:
    if np.isfinite(ed_vol) and np.isfinite(es_vol) and ed_vol > 0:
        return (ed_vol - es_vol) / ed_vol * 100.0
    return float('nan')


def ef_al(masks: np.ndarray, lv_label: int = 2) -> float:
    """Single-view area-length ejection fraction (EF_AL), as a percentage.

    masks: LV segmentation sequence of shape (T, H, W).
    """
    (ed_area, l_ed), (es_area, l_es) = _ed_es_measurements(masks, lv_label)
    ed_vol = _single_plane_volume(ed_area, l_ed)
    es_vol = _single_plane_volume(es_area, l_es)
    return _ef_percent(ed_vol, es_vol)


def ef_al_biplane(masks_2ch: np.ndarray, masks_4ch: np.ndarray, lv_label: int = 2) -> float:
    """Biplane area-length ejection fraction (EF_AL_Biplane), as a percentage.

    masks_2ch / masks_4ch: LV segmentation sequences of shape (T, H, W) for the
    2CH and 4CH views. ED/ES are selected independently per view.
    """
    (A2_ed, L2_ed), (A2_es, L2_es) = _ed_es_measurements(masks_2ch, lv_label)
    (A4_ed, L4_ed), (A4_es, L4_es) = _ed_es_measurements(masks_4ch, lv_label)
    ed_vol = _biplane_volume(A2_ed, L2_ed, A4_ed, L4_ed)
    es_vol = _biplane_volume(A2_es, L2_es, A4_es, L4_es)
    return _ef_percent(ed_vol, es_vol)
