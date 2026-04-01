# 🥇 Stanford RNA 3D Folding Part 2 — 6th Place Solution

**Competition:** [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)  
**Final standing:** 6th / 1867 teams · Score: 0.479 · Gold Medal 🏅

---

## Solution Overview

The core idea: use **Template-Based Modeling (TBM)** to generate cheap structural hypotheses, then use two neural models (**Protenix** and **RNAPro**) to independently refine them — feeding TBM and Protenix outputs *back into RNAPro as precomputed templates*. This cross-pollination strategy gave five structurally diverse predictions per target, which the competition metric (best-of-5 TM-score) rewards heavily.
```
                    ┌──────────────┐
                    │  Test Seqs   │
                    └──────┬───────┘
                           │
           ┌───────────────┼────────────────┐
           ▼               ▼                ▼
     ┌───────────┐   ┌───────────┐   (seq ≤ 1000 nt)
     │    TBM    │   │  Protenix │
     │  (fast)   │   │ (neural)  │
     └─────┬─────┘   └─────┬─────┘
           │               │
           └───────┬────────┘
                   ▼
            ┌─────────────┐
            │   RNAPro    │  ← conditioned on TBM/Protenix templates
            │  (refine)   │
            └─────┬───────┘
                  ▼
     ┌─────────────────────────────┐
     │        5 Predictions        │
     │  P1: RNAPro + TBM[0]        │
     │  P2: RNAPro + TBM[1]        │
     │  P3: RNAPro + Protenix[0]   │
     │  P4: RNAPro + Protenix[1]   │
     │  P5: Pure Protenix[0]       │
     └─────────────────────────────┘
```

---

## Detailed Approach

### Phase 1 — Template-Based Modeling (TBM)

A classical bioinformatics approach used as both a fast baseline and a source of structural templates for the neural stage.

**Template search** uses BioPython's `PairwiseAligner` in global mode with strong gap penalties to prevent residue-numbering drift.

**Coordinate transfer** maps C1' pseudoatom positions from the best-matching training structures onto query residues via the pairwise alignment, with linear interpolation for unmatched gaps.

**Geometry refinement** (`adaptive_rna_constraints`) applies within each chain segment:
- Bond constraint: i↔i+1 → ~5.95 Å
- Soft angle constraint: i↔i+2 → ~10.20 Å
- Laplacian smoothing to remove kinks
- Steric self-avoidance for longer chains (L ≥ 25)

**Diversity** across the 5 TBM predictions:

| Pred | Transform |
|------|-----------|
| 0 | Best template, no perturbation |
| 1 | Mild Gaussian noise (σ ∝ 1 − similarity) |
| 2 | Hinge rotation on longest chain segment |
| 3 | Independent rigid-body jitter per chain |
| 4 | Smooth low-frequency deformation (wiggle) |

### Phase 2 — Protenix

[Protenix](https://github.com/bytedance/Protenix) (ByteDance, AlphaFold3-style) run in **no-MSA, no-template** mode for sequences ≤ 1000 nt. The first 2 of 5 samples are forwarded to RNAPro as templates.

### Phase 3 — RNAPro Hybrid

[RNAPro](https://kaggle.com/datasets/theoviel/rnapro-src) is conditioned on precomputed C1' templates from both TBM and Protenix, with MSA + RibonanzaNet2 embeddings, 10 recycling cycles, and 200 diffusion steps.

**Final 5 predictions per target:**

| # | Description |
|---|-------------|
| 1 | RNAPro refined with TBM template 0 |
| 2 | RNAPro refined with TBM template 1 |
| 3 | RNAPro refined with Protenix template 0 |
| 4 | RNAPro refined with Protenix template 1 |
| 5 | Pure Protenix pred 1 (unrefined) |

> ⚠️ **Long sequence fallback:** For sequences longer than 1000 nt, all 5 predictions
> come directly from TBM (no Protenix, no RNAPro). The neural models are skipped
> entirely due to memory and time constraints.

---

## Repository Structure
```
stanford-rna-folding/
├── run_inference.py
├── configs/
│   └── inference_config.yaml
├── src/
│   ├── tbm/
│   │   ├── alignment.py
│   │   ├── constraints.py
│   │   ├── diversity.py
│   │   └── predict.py
│   ├── protenix/
│   │   └── predict.py
│   ├── rnapro/
│   │   └── predict.py
│   └── utils/
│       ├── data.py
│       └── submission.py
├── notebooks/
│   └── README.md
├── requirements.txt
└── LICENSE
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download external models

**Protenix:**
```bash
git clone https://github.com/bytedance/Protenix
# Place checkpoint at models/protenix/
```

**RNAPro:** [`theoviel/rnapro-src`](https://www.kaggle.com/datasets/theoviel/rnapro-src)  
Place checkpoint at `models/rnapro-private-best-500m.ckpt`

### 3. Configure paths

Edit `configs/inference_config.yaml` to point to your data, MSA files, and model checkpoints.

### 4. Run
```bash
# Full pipeline
python run_inference.py --config configs/inference_config.yaml

# Debug on 5 sequences
python run_inference.py --config configs/inference_config.yaml --debug
```

---

## Key Design Decisions

- **TBM → neural** rather than submitting TBM directly; RNAPro corrects geometry while retaining template structure
- **Protenix → RNAPro** cross-pollination introduces diversity neither model produces alone
- **Pure Protenix as Pred 5** provides an uncorrelated fallback for well-studied motifs
- **Chain-aware constraints** — refinement applied within each chain segment independently, avoiding phantom bonds across chain breaks

---

## Hardware

- 2 × NVIDIA T4 (Kaggle)

---

## Acknowledgements

- [Protenix](https://github.com/bytedance/Protenix) — ByteDance
- [RNAPro](https://www.kaggle.com/datasets/theoviel/rnapro-src) — Théo Viel
- [RibonanzaNet2](https://www.kaggle.com/models/shujun717/ribonanzanet2) — shujun717

---

## License

MIT — see [LICENSE](LICENSE).