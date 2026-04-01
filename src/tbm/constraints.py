"""
tbm/constraints.py
─────────────────────────────────────────────────────────────
Physics-inspired geometry refinement for RNA C1' coordinates.

Applies within each chain segment independently — no phantom
bonds across chain breaks in multi-chain assemblies.
"""

import numpy as np


def adaptive_rna_constraints(coordinates: np.ndarray, target_id: str,
                              segs_map: dict, confidence: float = 1.0,
                              passes: int = 2) -> np.ndarray:
    """
    Refine C1' geometry using soft physical constraints.

    Constraints applied (all vectorized, per chain segment):
      1. Bond i↔i+1  → ~5.95 Å
      2. Angle i↔i+2 → ~10.20 Å  (soft)
      3. Laplacian smoothing (removes kinks)
      4. Light steric self-avoidance (prevents collapse, L ≥ 25)

    Correction strength scales with (1 - confidence): high-confidence
    templates receive minimal nudging; low-confidence ones are corrected
    more aggressively.
    """
    coords = coordinates.copy()
    segments = segs_map.get(target_id, [(0, len(coords))])

    strength = 0.75 * (1.0 - min(confidence, 0.97))
    strength = max(strength, 0.02)

    for _ in range(passes):
        for (s, e) in segments:
            X = coords[s:e]
            L = e - s
            if L < 3:
                continue

            # (1) Bond i,i+1
            d = X[1:] - X[:-1]
            dist = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
            adj = d * ((5.95 - dist) / dist) * (0.22 * strength)
            X[:-1] -= adj
            X[1:]  += adj

            # (2) Soft i,i+2
            d2 = X[2:] - X[:-2]
            dist2 = np.linalg.norm(d2, axis=1, keepdims=True) + 1e-6
            adj2 = d2 * ((10.2 - dist2) / dist2) * (0.10 * strength)
            X[:-2] -= adj2
            X[2:]  += adj2

            # (3) Laplacian smoothing
            lap = 0.5 * (X[:-2] + X[2:]) - X[1:-1]
            X[1:-1] += (0.06 * strength) * lap

            # (4) Steric self-avoidance
            if L >= 25:
                k = min(L, 160) if L > 220 else L
                idx = np.linspace(0, L - 1, k).astype(int) if k < L else np.arange(L)
                P = X[idx]
                diff = P[:, None] - P[None, :]
                distm = np.linalg.norm(diff, axis=2) + 1e-6
                sep = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (distm < 3.2)
                if np.any(mask):
                    force = (3.2 - distm) / distm
                    vec = (diff * force[:, :, None] * mask[:, :, None]).sum(axis=1)
                    X[idx] += (0.015 * strength) * vec

            coords[s:e] = X

    return coords