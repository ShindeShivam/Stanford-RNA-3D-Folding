"""
protenix/predict.py
─────────────────────────────────────────────────────────────
Protenix (ByteDance AlphaFold3-style) inference wrapper.

Runs N_sample predictions per sequence (no MSA, no templates)
and extracts C1' coordinates from the raw all-atom output.

Sequences longer than `max_seq_len` are truncated before inference;
coordinates are padded back to the full sequence length with zeros.
"""

import gc
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch


def build_input_json(df, output_path: str, max_seq_len: int = 512) -> str:
    """Serialise a test DataFrame to the Protenix input JSON format."""
    data = [
        {
            "name": row["target_id"],
            "covalent_bonds": [],
            "sequences": [{
                "rnaSequence": {
                    "sequence": row["sequence"][:max_seq_len],
                    "count": 1,
                }
            }],
        }
        for _, row in df.iterrows()
    ]
    with open(output_path, "w") as f:
        json.dump(data, f)
    return output_path


def _extract_c1_coords(raw_coords: torch.Tensor, feat: dict,
                       full_seq_len: int) -> np.ndarray:
    """
    Extract C1' atom coordinates from full all-atom Protenix output.

    Uses centre_atom_mask if available, otherwise falls back to
    atom_to_tokatom_idx (indices 11 or 12) — whichever count is
    closest to the full sequence length.

    Returns (N_sample, seq_len, 3) float32 array.
    """
    if "centre_atom_mask" in feat:
        mask = (feat["centre_atom_mask"] == 1).to(raw_coords.device)
    elif "atom_to_tokatom_idx" in feat:
        m11 = (feat["atom_to_tokatom_idx"] == 11).to(raw_coords.device)
        m12 = (feat["atom_to_tokatom_idx"] == 12).to(raw_coords.device)
        mask = m11 if abs(m11.sum() - full_seq_len) <= abs(m12.sum() - full_seq_len) else m12
    else:
        mask = torch.zeros(raw_coords.shape[1], dtype=torch.bool,
                           device=raw_coords.device)

    coords = raw_coords[:, mask, :].detach().cpu().numpy()  # (N_sample, L, 3)

    # Pad / trim to full sequence length
    if coords.shape[1] != full_seq_len:
        padded = np.zeros((coords.shape[0], full_seq_len, 3), dtype=np.float32)
        n = min(coords.shape[1], full_seq_len)
        padded[:, :n] = coords[:, :n]
        coords = padded

    return coords


def run_protenix(df, cfg: dict, work_dir: str = "/tmp/protenix") -> dict:
    """
    Run Protenix inference on all sequences in `df`.

    Parameters
    ----------
    df       : DataFrame with 'target_id' and 'sequence' columns
    cfg      : protenix config sub-dict from inference_config.yaml
    work_dir : scratch directory for intermediate files

    Returns
    -------
    dict  target_id → np.ndarray (N_sample, seq_len, 3), or None on failure
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    protenix_dir = cfg["model_dir"]

    if protenix_dir not in sys.path:
        sys.path.insert(0, protenix_dir)
    os.environ["PROTENIX_ROOT_DIR"] = protenix_dir
    os.environ["LAYERNORM_TYPE"]    = "torch"

    from configs.configs_base      import configs as configs_base
    from configs.configs_data      import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config    import parse_configs
    from protenix.data.inference.infer_dataloader import InferenceDataset
    from runner.inference import (
        InferenceRunner, update_gpu_compatible_configs, update_inference_configs,
    )

    model_name = cfg["model_name"]
    n_samples  = cfg["n_samples"]
    max_len    = cfg["max_seq_len"]

    input_json = build_input_json(df, os.path.join(work_dir, "input.json"), max_len)

    base_cfg = {**configs_base, **{"data": data_configs}, **inference_configs}

    def _deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict):
                _deep_update(t[k], v)
            else:
                t[k] = v

    _deep_update(base_cfg, model_configs[model_name])

    arg_str = (
        f"--model_name {model_name} "
        f"--input_json_path {input_json} "
        f"--dump_dir {os.path.join(work_dir, 'outputs')} "
        f"--use_msa {'true' if cfg.get('use_msa') else 'false'} "
        f"--use_template {'true' if cfg.get('use_template') else 'false'} "
        f"--use_rna_msa false "
        f"--sample_diffusion.N_sample {n_samples} "
        f"--seeds {cfg.get('seed', 42)}"
    )

    ptx_cfg = parse_configs(configs=base_cfg, arg_str=arg_str,
                            fill_required_with_null=True)
    ptx_cfg = update_gpu_compatible_configs(ptx_cfg)

    runner  = InferenceRunner(ptx_cfg)
    dataset = InferenceDataset(ptx_cfg)
    seq_map = dict(zip(df["target_id"], df["sequence"]))
    results: dict = {}

    for i in range(len(dataset)):
        data, atom_array, err = dataset[i]
        tid      = data.get("sample_name", f"sample_{i}")
        full_seq = seq_map.get(tid, "")

        if err:
            print(f"  {tid}: data error — {err}")
            results[tid] = None
            del data, atom_array
            gc.collect(); torch.cuda.empty_cache()
            continue

        try:
            new_cfg = update_inference_configs(ptx_cfg, data["N_token"].item())
            new_cfg.sample_diffusion.N_sample = n_samples
            runner.update_model_configs(new_cfg)

            pred       = runner.predict(data)
            raw_coords = pred["coordinate"]
            coords     = _extract_c1_coords(
                raw_coords, data["input_feature_dict"], len(full_seq)
            )

            # Sanity: collapsed output check
            if coords.shape[1] > 1:
                diffs = np.linalg.norm(coords[0, 1:] - coords[0, :-1], axis=-1)
                if np.all(diffs < 1e-4):
                    print(f"  WARNING {tid}: collapsed coords — zeroing")
                    coords = np.zeros_like(coords)

            results[tid] = coords
            print(f"  {tid}: ✓  shape={coords.shape}")

        except Exception as exc:
            print(f"  {tid}: FAILED — {exc}")
            results[tid] = None

        finally:
            for var in ("pred", "raw_coords", "mask"):
                try: del locals()[var]
                except KeyError: pass
            del data, atom_array
            gc.collect(); torch.cuda.empty_cache()

    del runner, dataset
    gc.collect(); torch.cuda.empty_cache()
    return results


__all__ = ["run_protenix", "build_input_json"]