"""
tbm/alignment.py
─────────────────────────────────────────────────────────────
Template-Based Modeling: sequence alignment and coordinate transfer.
"""

import numpy as np
from Bio.Align import PairwiseAligner


def build_aligner() -> PairwiseAligner:
    """
    Build a global PairwiseAligner tuned for RNA structure transfer.

    Strong gap penalties discourage 'sliding', which would misplace
    residue numbering and corrupt coordinate transfer.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1.5

    for attr in [
        "open_gap_score", "query_left_open_gap_score",
        "query_right_open_gap_score", "target_left_open_gap_score",
        "target_right_open_gap_score",
    ]:
        setattr(aligner, attr, -8)

    for attr in [
        "extend_gap_score", "query_left_extend_gap_score",
        "query_right_extend_gap_score", "target_left_extend_gap_score",
        "target_right_extend_gap_score",
    ]:
        setattr(aligner, attr, -0.4)

    return aligner


# Module-level aligner instance (shared across calls)
_ALIGNER = build_aligner()


def find_similar_sequences(query_seq: str, train_seqs_df, train_coords_dict: dict,
                           top_n: int = 30) -> list:
    """
    Return the top_n training sequences most similar to query_seq.

    Uses a fast length pre-filter (±30%) before running alignment scoring.

    Returns
    -------
    list of (target_id, train_seq, normalized_score, coords_array)
        sorted by score descending.
    """
    similar = []
    for _, row in train_seqs_df.iterrows():
        tid, train_seq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict:
            continue
        if abs(len(train_seq) - len(query_seq)) / max(len(train_seq), len(query_seq)) > 0.3:
            continue
        raw = _ALIGNER.score(query_seq, train_seq)
        norm = raw / (2 * min(len(query_seq), len(train_seq)))
        similar.append((tid, train_seq, norm, train_coords_dict[tid]))

    similar.sort(key=lambda x: x[2], reverse=True)
    return similar[:top_n]


def adapt_template_to_query(query_seq: str, template_seq: str,
                             template_coords: np.ndarray) -> np.ndarray:
    """
    Map template C1' coordinates onto the query sequence via pairwise alignment.

    Unmatched query positions are filled by linear interpolation (or
    linear extrapolation at the termini, spaced 3 Å apart).

    Parameters
    ----------
    query_seq       : query RNA sequence (A/C/G/U)
    template_seq    : template RNA sequence
    template_coords : (L_template, 3) C1' coordinate array

    Returns
    -------
    new_coords : (L_query, 3) float64 array — no NaNs
    """
    alignment = next(iter(_ALIGNER.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)

    for (q_start, q_end), (t_start, t_end) in zip(*alignment.aligned):
        chunk = template_coords[t_start:t_end]
        if len(chunk) == (q_end - q_start):
            new_coords[q_start:q_end] = chunk

    # ── Interpolation / extrapolation for gaps ────────────────
    for i in range(len(new_coords)):
        if not np.isnan(new_coords[i, 0]):
            continue
        prev = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
        nxt  = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
        if prev >= 0 and nxt >= 0:
            w = (i - prev) / (nxt - prev)
            new_coords[i] = (1 - w) * new_coords[prev] + w * new_coords[nxt]
        elif prev >= 0:
            new_coords[i] = new_coords[prev] + [3.0, 0.0, 0.0]
        elif nxt >= 0:
            new_coords[i] = new_coords[nxt] + [3.0, 0.0, 0.0]
        else:
            new_coords[i] = [i * 3.0, 0.0, 0.0]

    return np.nan_to_num(new_coords)