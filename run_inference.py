#!/usr/bin/env python3
"""
run_inference.py
─────────────────────────────────────────────────────────────
End-to-end inference pipeline for the Stanford RNA 3D Folding
competition.

Three-stage hybrid:
  Phase 1 – TBM (Template-Based Modeling):   fast baseline, 5 predictions
  Phase 2 – Protenix:                         neural, 5 predictions (seq ≤ 1000 nt)
  Phase 3 – RNAPro (hybrid):
      Pred 1  →  RNAPro + TBM template slot 0
      Pred 2  →  RNAPro + TBM template slot 1
      Pred 3  →  RNAPro + Protenix template slot 0
      Pred 4  →  RNAPro + Protenix template slot 1
      Pred 5  →  Pure Protenix pred 1 (no RNAPro)

Long sequences (> rnapro.max_seq_len) fall back to TBM×5.

Usage
-----
    python run_inference.py --config configs/inference_config.yaml

Optional overrides:
    --test_csv    path/to/test_sequences.csv
    --output_dir  path/to/outputs/
    --debug       run on first 5 sequences only
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

# ── src on path ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.data       import build_segments_map, process_labels
from utils.submission import (coords_to_dataframe, get_tbm_coords,
                               get_protenix_coords, build_template_csv,
                               save_submission)
from tbm.predict      import predict_rna_structures


# ── CLI ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stanford RNA 3D Folding — inference")
    p.add_argument("--config",     default="configs/inference_config.yaml")
    p.add_argument("--test_csv",   default=None, help="Override test CSV path")
    p.add_argument("--output_dir", default=None, help="Override output directory")
    p.add_argument("--debug",      action="store_true",
                   help="Run on first 5 sequences only")
    return p.parse_args()


# ── Phase 1: TBM ──────────────────────────────────────────────

def phase1_tbm(test_df, train_seqs, train_coords, segs_map, cfg, output_dir):
    print("\n" + "="*60)
    print("PHASE 1: Template-Based Modeling (TBM)")
    print("="*60)

    all_preds = []
    t0 = time.time()
    for idx, row in test_df.iterrows():
        if idx % 10 == 0:
            print(f"  [{idx}/{len(test_df)}]  {time.time()-t0:.0f}s")
        tid, seq = row["target_id"], row["sequence"]
        preds = predict_rna_structures(
            row, train_seqs, train_coords, segs_map,
            n_predictions=cfg["tbm"]["n_predictions"],
        )
        for j in range(len(seq)):
            res = {"ID": f"{tid}_{j+1}", "resname": seq[j], "resid": j+1}
            for i in range(5):
                res[f"x_{i+1}"], res[f"y_{i+1}"], res[f"z_{i+1}"] = preds[i][j]
            all_preds.append(res)

    sub = pd.DataFrame(all_preds)
    path = os.path.join(output_dir, "tbm_submission.csv")
    sub.to_csv(path, index=False)
    print(f"✓ TBM done → {path}")
    return sub


# ── Phase 2: Protenix ─────────────────────────────────────────

def phase2_protenix(test_df, cfg, output_dir, work_dir):
    from protenix.predict import run_protenix

    print("\n" + "="*60)
    print("PHASE 2: Protenix")
    print("="*60)

    ptx_cfg  = cfg["protenix"]
    ptx_df   = test_df[test_df["sequence"].str.len() <= ptx_cfg["max_seq_len"]].reset_index(drop=True)
    print(f"  Sequences ≤ {ptx_cfg['max_seq_len']} nt: {len(ptx_df)} / {len(test_df)}")

    preds = run_protenix(ptx_df, ptx_cfg, work_dir=os.path.join(work_dir, "protenix"))

    rows = []
    seq_map = dict(zip(ptx_df["target_id"], ptx_df["sequence"]))
    for tid, coords in preds.items():
        seq = seq_map.get(tid, "")
        if coords is None or not seq:
            continue
        for i in range(len(seq)):
            row = {"ID": f"{tid}_{i+1}", "resname": seq[i], "resid": i+1}
            for s in range(coords.shape[0]):
                row[f"x_{s+1}"] = float(coords[s, i, 0])
                row[f"y_{s+1}"] = float(coords[s, i, 1])
                row[f"z_{s+1}"] = float(coords[s, i, 2])
            rows.append(row)

    ptx_out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "protenix_preds.csv")
    ptx_out.to_csv(path, index=False)
    print(f"✓ Protenix done → {path}  ({len(ptx_out):,} rows)")
    return ptx_out


# ── Phase 3: RNAPro hybrid ────────────────────────────────────

def phase3_rnapro(test_df, tbm_sub, ptx_df, cfg, output_dir, work_dir):
    import argparse as _argparse
    from rnapro.utils.seed import seed_everything
    from rnapro.config.config import ConfigManager, ArgumentNotSet
    from configs.configs_base      import configs as configs_base
    from configs.configs_data      import data_configs
    from configs.configs_inference import inference_configs

    from rnapro.predict import run_rnapro_single, load_rnapro_model

    print("\n" + "="*60)
    print("PHASE 3: RNAPro hybrid inference")
    print("="*60)

    # ── Build 4-slot template .pt ─────────────────────────────
    template_csv = os.path.join(output_dir, "four_templates.csv")
    build_template_csv(tbm_sub, ptx_df, template_csv)

    rnapro_dir = os.path.dirname(cfg["rnapro"]["checkpoint"])
    sys.path.insert(0, os.path.join(rnapro_dir, "RNAPro"))
    os.chdir(os.path.join(rnapro_dir, "RNAPro"))
    os.system(
        f"python preprocess/convert_templates_to_pt_files.py "
        f"--input_csv {template_csv} --output_name four_templates.pt"
    )
    os.chdir(work_dir)

    template_pt = os.path.join(rnapro_dir, "RNAPro", "release_data",
                                "kaggle", "four_templates.pt")

    # ── RNAPro config ─────────────────────────────────────────
    rc = cfg["rnapro"]
    all_cfgs = {**configs_base, **{"data": data_configs}, **inference_configs}
    manager  = ConfigManager(all_cfgs, fill_required_with_null=True)

    arg_string = f"""
    --model_name rnapro_base
    --load_checkpoint_path {rc['checkpoint']}
    --load_strict true
    --dtype {rc.get('dtype', 'bf16')}
    --use_template ca_precomputed
    --model.use_template ca_precomputed
    --template_data {template_pt}
    --use_msa {str(rc.get('use_msa', True)).lower()}
    --rna_msa_dir {cfg['data']['msa_dir']}
    --model.use_RibonanzaNet2 true
    --model.ribonanza_net_path {rc.get('ribonanza_net_path', '')}
    --model.template_embedder.n_blocks 2
    --model.N_cycle {rc.get('n_cycles', 10)}
    --sample_diffusion.N_sample {rc.get('n_samples', 1)}
    --sample_diffusion.N_step {rc.get('n_diffusion_steps', 200)}
    --seeds {rc.get('seed', 42)}
    --dump_dir {os.path.join(work_dir, 'rnapro_outputs')}
    --num_workers 0
    --triangle_attention torch
    --triangle_multiplicative torch
    --deterministic false
    """.strip().split()

    p2 = _argparse.ArgumentParser()
    p2.add_argument("--max_len", type=int, default=1000)
    for key, (dtype, *_) in manager.config_infos.items():
        p2.add_argument("--" + key, type=str, default=ArgumentNotSet(), required=False)
    args = vars(p2.parse_args(arg_string))
    rna_cfg = manager.merge_configs(args)
    rna_cfg.max_len = rc.get("max_seq_len", 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_rnapro_model(rna_cfg, rc["checkpoint"], device)

    # ── Per-target inference ──────────────────────────────────
    results = []
    for idx, row in test_df.iterrows():
        tid, seq = row["target_id"], row["sequence"]
        N = len(seq)
        print(f"\n[{idx+1}/{len(test_df)}] {tid} ({N} nt)", end="  ")

        tbm_all    = get_tbm_coords(tid, tbm_sub)
        ptx_coords = get_protenix_coords(tid, ptx_df)
        coords_list = []

        # Long sequences: pure TBM fallback
        if N > rna_cfg.max_len:
            print("→ TBM×5 (too long)")
            results.append(coords_to_dataframe(seq, tbm_all[:5], tid))
            continue

        print("→ RNAPro hybrid")

        # Pred 1-4: RNAPro with template slots 0,1,2,3
        for slot in range(4):
            label = f"TBM slot {slot}" if slot < 2 else f"Protenix slot {slot}"
            print(f"  [Pred {slot+1}] RNAPro+{label}...", end=" ")
            c = run_rnapro_single(seq, tid, slot, template_pt, rna_cfg,
                                   model, device, work_dir)
            if c is not None:
                coords_list.append(c)
                print("✓")
            else:
                fallback = tbm_all[slot] if slot < 2 else (
                    ptx_coords[slot - 2] if ptx_coords else tbm_all[slot]
                )
                coords_list.append(fallback)
                print("✗ (fallback)")
            torch.cuda.empty_cache()

        # Pred 5: pure Protenix
        print("  [Pred 5] Pure Protenix...", end=" ")
        coords_list.append(ptx_coords[0] if ptx_coords else tbm_all[4])
        print("✓" if ptx_coords else "✗ (TBM fallback)")

        results.append(coords_to_dataframe(seq, coords_list, tid))
        torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────
    sub_cfg = cfg["submission"]
    return save_submission(
        results,
        os.path.join(output_dir, "submission.csv"),
        clip_min=sub_cfg["coord_clip_min"],
        clip_max=sub_cfg["coord_clip_max"],
    )


# ── Main ───────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.test_csv:
        cfg["data"]["test_sequences"] = args.test_csv
    if args.output_dir:
        cfg["data"]["output_dir"] = args.output_dir

    output_dir = cfg["data"]["output_dir"]
    work_dir   = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────
    print("Loading data...")
    train_seqs   = pd.read_csv(cfg["data"]["train_sequences"])
    train_labels = pd.read_csv(cfg["data"]["train_labels"])
    test_df      = pd.read_csv(cfg["data"]["test_sequences"])

    if args.debug:
        test_df = test_df.head(5).reset_index(drop=True)
        print("DEBUG mode: running on first 5 sequences")

    train_coords = process_labels(train_labels)
    segs_map, _  = build_segments_map(test_df)

    # ── Pipeline ───────────────────────────────────────────────
    tbm_sub = phase1_tbm(test_df, train_seqs, train_coords, segs_map, cfg, output_dir)
    ptx_df  = phase2_protenix(test_df, cfg, output_dir, work_dir)

    # Clean up Protenix from sys.modules before loading RNAPro
    mods = [k for k in sys.modules if any(x in k for x in ["protenix", "configs", "runner"])]
    for m in mods:
        del sys.modules[m]
    torch.cuda.empty_cache(); gc.collect()

    phase3_rnapro(test_df, tbm_sub, ptx_df, cfg, output_dir, work_dir)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()