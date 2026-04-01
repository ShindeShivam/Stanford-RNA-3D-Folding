"""
utils/submission.py
─────────────────────────────────────────────────────────────
Helpers for assembling and validating the final submission CSV.
"""

import numpy as np
import pandas as pd


def coords_to_dataframe(sequence: str, coords_list: list,
                         target_id: str) -> pd.DataFrame:
    """
    Convert a list of 5 coordinate arrays into a submission-format DataFrame.

    Parameters
    ----------
    sequence    : RNA sequence string (length N)
    coords_list : list of 5 np.ndarray, each shape (N, 3)
    target_id   : competition target identifier

    Returns
    -------
    DataFrame with columns: ID, resname, resid, x_1..z_5
    """
    rows = []
    for i, residue in enumerate(sequence):
        row = {"ID": f"{target_id}_{i+1}", "resname": residue, "resid": i+1}
        for j, coords in enumerate(coords_list):
            row[f"x_{j+1}"] = float(coords[i, 0])
            row[f"y_{j+1}"] = float(coords[i, 1])
            row[f"z_{j+1}"] = float(coords[i, 2])
        rows.append(row)
    return pd.DataFrame(rows)


def get_tbm_coords(target_id: str, tbm_sub: pd.DataFrame) -> list:
    """Extract all 5 TBM coordinate arrays for a target from a submission CSV."""
    rows = tbm_sub[tbm_sub["ID"].str.startswith(target_id + "_")]
    return [rows[[f"x_{i}", f"y_{i}", f"z_{i}"]].values for i in range(1, 6)]


def get_protenix_coords(target_id: str, ptx_df: pd.DataFrame):
    """
    Return [coords_sample_0, coords_sample_1] for a target from the
    Protenix predictions CSV, or None if the target is not present.
    """
    rows = ptx_df[ptx_df["ID"].str.startswith(target_id + "_")]
    if len(rows) == 0:
        return None
    return [
        rows[["x_1", "y_1", "z_1"]].values,
        rows[["x_2", "y_2", "z_2"]].values,
    ]


def build_template_csv(tbm_df: pd.DataFrame, ptx_df: pd.DataFrame,
                        output_path: str) -> pd.DataFrame:
    """
    Merge TBM (slots 0-1) and Protenix (slots 2-3) predictions into a
    single template CSV for RNAPro's precomputed-template converter.

    A dummy slot 4 (zeros) is appended to satisfy the converter's
    expectation of exactly 5 slots.

    Parameters
    ----------
    tbm_df      : full TBM submission DataFrame (5 predictions)
    ptx_df      : Protenix predictions DataFrame (2 predictions)
    output_path : where to write the merged CSV

    Returns
    -------
    Merged DataFrame (also written to output_path)
    """
    tbm_sub = tbm_df[["ID", "resname", "resid",
                       "x_1", "y_1", "z_1",
                       "x_2", "y_2", "z_2"]].copy()

    ptx_sub = ptx_df[["ID", "x_1", "y_1", "z_1",
                            "x_2", "y_2", "z_2"]].rename(columns={
        "x_1": "x_3", "y_1": "y_3", "z_1": "z_3",
        "x_2": "x_4", "y_2": "y_4", "z_2": "z_4",
    })

    merged = pd.merge(tbm_sub, ptx_sub, on="ID", how="left")
    merged["x_5"] = 0.0
    merged["y_5"] = 0.0
    merged["z_5"] = 0.0

    merged.to_csv(output_path, index=False)
    return merged


def save_submission(results: list, output_path: str,
                    clip_min: float = -999.999,
                    clip_max: float = 9999.999) -> pd.DataFrame:
    """
    Concatenate per-target DataFrames, clip coordinate values, and save.

    Returns the final submission DataFrame.
    """
    col_order = (["ID", "resname", "resid"]
                 + [f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]])
    sub = pd.concat(results, ignore_index=True)
    coord_cols = [c for c in col_order if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(clip_min, clip_max)
    sub[col_order].to_csv(output_path, index=False)
    print(f"✓ Saved {output_path}  ({len(sub):,} rows)")
    return sub


__all__ = [
    "coords_to_dataframe", "get_tbm_coords", "get_protenix_coords",
    "build_template_csv", "save_submission",
]