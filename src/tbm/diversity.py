"""
tbm/diversity.py
─────────────────────────────────────────────────────────────
Structural diversity transforms applied to TBM predictions.

These create structurally distinct conformations from a single
template, improving ensemble coverage for the competition metric
(which scores the best of 5 predictions).
"""

import numpy as np


# ── Rotation utilities ────────────────────────────────────────

def _rotmat(axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation matrix for `axis` by `ang` radians."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis) + 1e-12
    x, y, z = axis
    c, s, C = np.cos(ang), np.sin(ang), 1.0 - np.cos(ang)
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C  ],
    ], dtype=float)


# ── Transform functions ───────────────────────────────────────

def apply_hinge(coords: np.ndarray, seg: tuple, rng: np.random.Generator,
                max_angle_deg: float = 25) -> np.ndarray:
    """
    Rotate the C-terminal half of a chain segment around a random pivot.

    Mimics a hinge-like domain motion. Skipped for segments shorter than 30.
    """
    s, e = seg
    L = e - s
    if L < 30:
        return coords

    pivot = s + int(rng.integers(10, L - 10))
    axis  = rng.normal(size=3)
    ang   = np.deg2rad(float(rng.uniform(-max_angle_deg, max_angle_deg)))
    R     = _rotmat(axis, ang)

    X = coords.copy()
    p0 = X[pivot].copy()
    X[pivot + 1:e] = (X[pivot + 1:e] - p0) @ R.T + p0
    return X


def jitter_chains(coords: np.ndarray, segments: list, rng: np.random.Generator,
                  max_angle_deg: float = 12, max_trans: float = 1.5) -> np.ndarray:
    """
    Apply independent random rigid-body transforms to each chain.

    Introduces inter-chain diversity while preserving intra-chain geometry.
    The whole assembly is recentered afterward to avoid coordinate drift.
    """
    X = coords.copy()
    global_center = X.mean(axis=0, keepdims=True)

    for (s, e) in segments:
        axis  = rng.normal(size=3)
        ang   = np.deg2rad(float(rng.uniform(-max_angle_deg, max_angle_deg)))
        R     = _rotmat(axis, ang)
        shift = rng.normal(size=3)
        shift = shift / (np.linalg.norm(shift) + 1e-12) * float(rng.uniform(0.0, max_trans))
        c     = X[s:e].mean(axis=0, keepdims=True)
        X[s:e] = (X[s:e] - c) @ R.T + c + shift

    X -= X.mean(axis=0, keepdims=True) - global_center
    return X


def smooth_wiggle(coords: np.ndarray, segments: list, rng: np.random.Generator,
                  amp: float = 0.8) -> np.ndarray:
    """
    Add smooth, low-frequency deformations interpolated from sparse control points.

    Produces natural-looking flexibility without breaking local geometry.
    Skipped for segments shorter than 20 residues.
    """
    X = coords.copy()
    for (s, e) in segments:
        L = e - s
        if L < 20:
            continue
        n_ctrl   = 6
        ctrl_x   = np.linspace(0, L - 1, n_ctrl)
        ctrl_d   = rng.normal(0, amp, size=(n_ctrl, 3))
        t        = np.arange(L)
        disp     = np.vstack([np.interp(t, ctrl_x, ctrl_d[:, k]) for k in range(3)]).T
        X[s:e]  += disp
    return X