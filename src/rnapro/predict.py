"""
rnapro/predict.py
─────────────────────────────────────────────────────────────
RNAPro inference wrapper with precomputed C1' template support.

Each call predicts one structure using a specific template slot
from a pre-built .pt template file.  Run once per slot (0-3).
"""

import json
import os
import sys

import numpy as np
import torch


def create_input_json(sequence: str, target_id: str, output_path: str) -> str:
    """Write a single-sequence RNAPro input JSON."""
    data = [{
        "sequences": [{"rnaSequence": {"sequence": sequence, "count": 1}}],
        "name": target_id,
    }]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path


def run_rnapro_single(sequence: str, target_id: str, template_idx: int,
                      template_pt_path: str, cfg, model,
                      device: torch.device, work_dir: str) -> np.ndarray | None:
    """
    Run one RNAPro prediction for `target_id` using `template_idx`.

    Extracts C1' atoms by name from the all-atom output.

    Parameters
    ----------
    sequence         : full RNA sequence (A/C/G/U)
    target_id        : target identifier
    template_idx     : which slot from the template .pt file to condition on
    template_pt_path : path to precomputed template .pt file
    cfg              : RNAPro config object (mutable — input_json_path is set)
    model            : loaded RNAPro model (eval mode)
    device           : torch device
    work_dir         : scratch dir for per-call JSON files

    Returns
    -------
    np.ndarray (seq_len, 3) or None on failure
    """
    from rnapro.data.infer_data_pipeline import get_inference_dataloader
    from rnapro.utils.torch_utils import to_device

    os.makedirs(os.path.join(work_dir, "inputs"), exist_ok=True)
    json_path = os.path.join(work_dir, "inputs", f"{target_id}_t{template_idx}.json")
    create_input_json(sequence, target_id, json_path)

    cfg.input_json_path = json_path
    cfg.template_data   = template_pt_path
    cfg.template_idx    = template_idx

    try:
        dataloader = get_inference_dataloader(configs=cfg)
        for batch in dataloader:
            data, atom_array, error_msg = batch[0]

            if error_msg:
                raise RuntimeError(f"Data error: {error_msg}")

            data = to_device(data, device)
            prec = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=prec):
                prediction, _, _ = model(
                    input_feature_dict=data["input_feature_dict"],
                    label_full_dict=None,
                    label_dict=None,
                    mode="inference",
                )

            coords_tensor = prediction["coordinate"]
            if coords_tensor.dim() == 3:
                coords_tensor = coords_tensor.squeeze(0)
            atom_positions = coords_tensor.cpu().numpy()  # (all_atoms, 3)

            if atom_positions.shape[0] <= 1:
                raise RuntimeError("Fold failure: single atom returned")

            # Extract C1' by atom name
            atom_names = np.char.strip(atom_array.atom_name.astype(str))
            c1_indices = np.where(atom_names == "C1'")[0]
            N = len(sequence)

            if len(c1_indices) >= N:
                coords = atom_positions[c1_indices[:N]]
            else:
                indices = np.linspace(0, atom_positions.shape[0] - 1, N, dtype=int)
                coords  = atom_positions[indices]

            return coords  # (N, 3)

    except Exception as exc:
        print(f"    RNAPro slot {template_idx} failed for {target_id}: {exc}")
        return None

    finally:
        torch.cuda.empty_cache()


def load_rnapro_model(cfg, checkpoint_path: str, device: torch.device):
    """Load and return an eval-mode RNAPro model."""
    from rnapro.model.RNAPro import RNAPro
    from rnapro.utils.seed import seed_everything

    seed_everything(42, deterministic=False)
    model = RNAPro(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)

    state = ckpt["model"]
    if list(state.keys())[0].startswith("module."):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"✓ RNAPro loaded ({n:,} parameters)")
    return model


__all__ = ["run_rnapro_single", "load_rnapro_model", "create_input_json"]