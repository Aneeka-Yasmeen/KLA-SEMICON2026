"""
Minimal, dependency-light array I/O helpers for run.py -- .npy only (this package's required
format), no Pillow dependency needed since no other image format is in scope here.
"""

import numpy as np


def infer_hw_from_shape(shape):
    """Given a 2D or 3D array shape, returns the (H, W) it will have after grayscale conversion,
    without needing the actual data -- shared by the fast shape-peek (batching) and the real
    loader below so they can never disagree about what shape a file will produce."""
    if len(shape) == 2:
        return shape
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4):
            return shape[0], shape[1]  # (H, W, C)
        if shape[0] in (1, 3, 4) and shape[-1] not in (1, 3, 4):
            return shape[1], shape[2]  # (C, H, W)
        if shape[-1] in (1, 3, 4):
            return shape[0], shape[1]  # ambiguous -- default to channel-last
    raise ValueError(f"unsupported array shape {shape}: expected 2D grayscale or 3D with a channel axis in {{1,3,4}}")


def _to_grayscale_3d(arr):
    """Converts a 3D array to 2D grayscale via standard luminance weights, not an arbitrary
    channel slice. Handles both channel-last (H,W,C) and channel-first (C,H,W)."""
    if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
        channel_last = arr
    elif arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        channel_last = np.moveaxis(arr, 0, -1)
    elif arr.shape[-1] in (1, 3, 4):
        channel_last = arr
    else:
        raise ValueError(
            f"unsupported 3D array shape {arr.shape}: no axis of size 1, 3, or 4 to treat as channels"
        )
    if channel_last.shape[-1] == 1:
        return channel_last[..., 0]
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (channel_last[..., :3] * weights).sum(axis=-1)


def load_npy_grayscale(path):
    """Loads a .npy file as a single-channel float32 (H, W) array. Handles 2D grayscale directly,
    3D channel-first/last arrays (converted via luminance, not sliced), and raw 0-255-scale data
    (auto-rescaled to [0,1] -- a genuinely [0,1]-ish array, even with intentional slight
    out-of-range excursions, never approaches raw 0-255 scale, so this is a safe heuristic)."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = _to_grayscale_3d(arr)
    elif arr.ndim != 2:
        raise ValueError(f"{path}: unsupported array shape {arr.shape} -- expected 2D or 3D with a channel axis")
    if arr.max() > 5.0:
        arr = arr / 255.0
    return arr


def sanitize_finite(arr):
    """Replaces any NaN/Inf with the array's own finite-value median (or 0.5 if none are finite),
    so a corrupted input still produces a real restoration attempt instead of propagating
    NaN/Inf through the model. Returns (clean_array, num_replaced)."""
    bad_mask = ~np.isfinite(arr)
    if not bad_mask.any():
        return arr, 0
    finite_vals = arr[~bad_mask]
    fallback = float(np.median(finite_vals)) if finite_vals.size else 0.5
    arr = arr.copy()
    arr[bad_mask] = fallback
    return arr, int(bad_mask.sum())
