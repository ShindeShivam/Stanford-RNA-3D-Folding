"""
utils/data.py
─────────────────────────────────────────────────────────────
Data loading utilities: FASTA parsing, stoichiometry parsing,
chain segment maps, and training coordinate extraction.
"""

import pandas as pd
import numpy as np


# ── FASTA / stoichiometry ─────────────────────────────────────

def parse_fasta(fasta_content: str) -> dict:
    """
    Minimal FASTA parser.  Returns {chain_id: sequence_string}.

    Handles the Kaggle extra/parse_fasta_py.py variant
    (which may return tuples) as well as plain strings.
    """
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur is not None:
                out[cur] = "".join(parts)
            cur, parts = line[1:].split()[0], []
        else:
            parts.append(line.replace(" ", ""))
    if cur is not None:
        out[cur] = "".join(parts)
    return out


def parse_stoichiometry(stoich: str) -> list:
    """
    Parse stoichiometry string like 'A:2;B:1' →  [('A', 2), ('B', 1)].
    Returns [] for null / empty input.
    """
    if pd.isna(stoich) or str(stoich).strip() == "":
        return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";")
            for ch, cnt in [part.split(":")]]


# ── Chain segment maps ────────────────────────────────────────

def get_chain_segments(row) -> list:
    """
    Return list of (start, end) index pairs for each chain copy
    within the full concatenated sequence.

    Falls back to a single [(0, len(seq))] segment if stoichiometry
    or all_sequences data are missing / inconsistent.
    """
    seq    = row["sequence"]
    stoich = row.get("stoichiometry", "")
    all_seq = row.get("all_sequences", "")

    if (pd.isna(stoich) or pd.isna(all_seq)
            or str(stoich).strip() == "" or str(all_seq).strip() == ""):
        return [(0, len(seq))]

    try:
        chain_dict = parse_fasta(all_seq)
        order      = parse_stoichiometry(stoich)
        segs, pos  = [], 0
        for ch, cnt in order:
            base = chain_dict.get(ch)
            if base is None:
                return [(0, len(seq))]
            for _ in range(cnt):
                segs.append((pos, pos + len(base)))
                pos += len(base)
        return [(0, len(seq))] if pos != len(seq) else segs
    except Exception:
        return [(0, len(seq))]


def build_segments_map(df: pd.DataFrame) -> tuple:
    """
    Build segment and stoichiometry maps for an entire sequences DataFrame.

    Returns
    -------
    seg_map    : dict  target_id → list of (start, end)
    stoich_map : dict  target_id → stoichiometry string
    """
    seg_map, stoich_map = {}, {}
    for _, r in df.iterrows():
        tid = r["target_id"]
        seg_map[tid]    = get_chain_segments(r)
        stoich_map[tid] = str(r.get("stoichiometry", "") or "")
    return seg_map, stoich_map


# ── Training labels ───────────────────────────────────────────

def process_labels(labels_df: pd.DataFrame) -> dict:
    """
    Build a dict mapping target_id → (L, 3) C1' coordinate array
    from the flat training labels CSV.
    """
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, group in labels_df.groupby(prefixes):
        coords[prefix] = (
            group.sort_values("resid")[["x_1", "y_1", "z_1"]].values
        )
    return coords


__all__ = [
    "parse_fasta", "parse_stoichiometry",
    "get_chain_segments", "build_segments_map",
    "process_labels",
]