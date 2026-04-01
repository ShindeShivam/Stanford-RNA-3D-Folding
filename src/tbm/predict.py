"""
tbm/predict.py
─────────────────────────────────────────────────────────────
Top-level TBM prediction: template selection, coordinate transfer,
diversity augmentation, and geometry refinement.
"""

import numpy as np

from .alignment import find_similar_sequences, adapt_template_to_query
from .constraints import adaptive_rna_constraints
from .diversity import apply_hinge, jitter_chains, smooth_wiggle


def predict_rna_structures(row, train_seqs_df, train_coords_dict: dict,
                            segs_map: dict, n_predictions: int = 5) -> list:
    """
    Generate `n_predictions` diverse C1' coordinate arrays for one target.

    Strategy
    --------
    Pred 0  — best template, no perturbation (highest fidelity)
    Pred 1  — mild Gaussian noise, scale inversely proportional to similarity
    Pred 2  — hinge rotation on the longest chain segment
    Pred 3  — independent rigid-body jitter per chain
    Pred 4  — smooth low-frequency deformation (wiggle)

    Templates for preds 1-4 are sampled from the top-12 pool with
    exponential weights biased toward high similarity, and de-weighted
    for already-used templates to maximise diversity.

    Falls back to a straight-line scaffold if no templates are found.

    Parameters
    ----------
    row               : DataFrame row with 'target_id' and 'sequence'
    train_seqs_df     : full training sequences DataFrame
    train_coords_dict : dict mapping target_id → C1' coords array
    segs_map          : dict mapping target_id → list of (start, end) segments
    n_predictions     : number of predictions to return (default 5)

    Returns
    -------
    list of np.ndarray, each shape (seq_len, 3)
    """
    tid      = row["target_id"]
    seq      = row["sequence"]
    segments = segs_map.get(tid, [(0, len(seq))])

    cands = find_similar_sequences(seq, train_seqs_df, train_coords_dict, top_n=30)
    predictions, used = [], set()

    for i in range(n_predictions):
        seed = (abs(hash(tid)) + i * 10007) % (2 ** 32)
        rng  = np.random.default_rng(seed)

        # ── Fallback: no templates ────────────────────────────
        if not cands:
            coords = np.zeros((len(seq), 3), dtype=float)
            for (s, e) in segments:
                for j in range(s + 1, e):
                    coords[j] = coords[j - 1] + [5.95, 0, 0]
            predictions.append(coords)
            continue

        # ── Template selection ────────────────────────────────
        if i == 0:
            t_id, t_seq, sim, t_coords = cands[0]
        else:
            K    = min(12, len(cands))
            sims = np.array([cands[k][2] for k in range(K)], float)
            w    = np.exp((sims - sims.max()) / 0.08)
            for k in range(K):
                if cands[k][0] in used:
                    w[k] *= 0.10
            w /= w.sum() + 1e-12
            k    = int(rng.choice(np.arange(K), p=w))
            t_id, t_seq, sim, t_coords = cands[k]

        used.add(t_id)

        # ── Coordinate transfer ───────────────────────────────
        adapted = adapt_template_to_query(seq, t_seq, t_coords)

        # ── Diversity transform ───────────────────────────────
        if i == 0:
            X = adapted
        elif i == 1:
            X = adapted + rng.normal(0, max(0.01, (0.40 - sim) * 0.06), adapted.shape)
        elif i == 2:
            longest = max(segments, key=lambda se: se[1] - se[0])
            X = apply_hinge(adapted, longest, rng, max_angle_deg=22)
        elif i == 3:
            X = jitter_chains(adapted, segments, rng, max_angle_deg=10, max_trans=1.0)
        else:
            X = smooth_wiggle(adapted, segments, rng, amp=0.7)

        refined = adaptive_rna_constraints(X, tid, segs_map, confidence=sim, passes=2)
        predictions.append(refined)

    return predictions


__all__ = ["predict_rna_structures"]