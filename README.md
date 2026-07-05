# phosphoswitch-md

MD simulation analysis for the LMNA Y45 phosphoswitch paper.

This package validates computationally that designed LMNA variant peptides
act as true phosphoswitches: in unbiased 600 ns GROMACS simulations, the
*unphosphorylated* (apo) variant preferentially occupies the predicted
straight-helix conformation, and the *phosphorylated* (phos) variant shifts
population toward the bent-hairpin conformation.

---

## Why MD is needed for this paper

Protein design produces sequences predicted (by AlphaFold3) to adopt two
distinct 3-D structures depending on whether Tyr45/Y30 is phosphorylated.
But AF3 predictions are single-structure outputs — they do not say whether
the designed sequence will actually *switch* in solution, how much energy
separates the two states, or whether unphosphorylated variants might collapse
into the bent state on their own.

**Unbiased all-atom MD simulation answers these questions directly:**

1. **Conformational stability**: Does each variant stay near its intended
   AF3-predicted reference structure, or does it collapse into a random coil?

2. **Phospho-driven switching**: Does adding the −2e phosphate (PTR30) shift
   the conformational equilibrium from the helix basin into the hairpin basin?

3. **Mechanism**: Which residues contact the phosphate in simulation?  This
   validates (or refutes) the designed electrostatic contacts.

4. **Ranking candidates**: Basin-occupancy enrichment (phos-MD in phos-state /
   apo-MD in phos-state) gives a quantitative switch quality score for
   choosing which variants to characterize experimentally.

---

## The LMNA Y45 construct

- **Target protein**: LMNA (Lamin A), residues Y45 ± 29 amino acids → 59 aa construct
- **Phosphosite**: Y30 in the construct (= Y45 in full-length LMNA)
- **Reference states**:
  - Apo reference: straight α-helix (AF3 prediction without phosphorylation)
  - Phos reference: bent hairpin (AF3 prediction with PTR30, dianionic −2e phospho-Tyr)
- **Simulations**: 3 replicates × 600 ns per condition (apo / phos) per candidate
- **Force field**: CHARMM36m (2022-Jul), TIP3P water, 150 mM NaCl, 310 K, 1 atm
- **System preparation**: CHARMM-GUI Solution Builder

---

## What basin occupancy means biologically

For a residue to function as a **phosphoswitch**, it must redirect the
conformational equilibrium rather than merely destabilize one state.
We quantify this using the **basin occupancy** framework:

Each simulation frame is assigned to the reference state it resembles most
closely, measured by Kabsch-aligned Cα RMSD:

```
basin(frame) = "apo"  if RMSD_to_apo_ref  < RMSD_to_phos_ref
             = "phos" if RMSD_to_phos_ref ≤ RMSD_to_apo_ref
```

We define two occupancy thresholds:
- **Strict (4 Å)**: The trajectory is tightly within a basin — clear
  structural convergence to the intended state.
- **Loose (7 Å)**: Includes transient visits and intermediate conformations.

The **phos-basin enrichment ratio**:

```
E = frac_phos_basin(phos MD) / frac_phos_basin(apo MD)
```

measures whether phosphorylation *causes* phos-state occupancy.

- **E >> 1**: Successful switch.  Phosphorylation drives population into the
  hairpin state.  apo MD stays in the helix; phos MD visits the hairpin.
- **E ≈ 1**: No switch.  Both conditions visit the phos state equally — the
  phosphate is not the deciding factor.
- **E < 1**: Inverted switch.  Phosphorylation drives the trajectory *away*
  from the phos state (unlikely but detectable).

---

## What a successful switch looks like in the data

A candidate that would be experimentally pursued satisfies all of:

| Criterion | Threshold | Meaning |
|---|---|---|
| `apo_MD_frac_apo_strict` | > 0.50 | Apo MD mostly stays in apo (helix) state |
| `phos_MD_frac_phos_loose` | > 0.20 | Phos MD visits hairpin substantially |
| `enrich_phos_loose` | > 2.0 | Phos MD ≥2× more phos-like than apo MD |
| `log2_phos_enrich_loose` | > +1.0 | log2 fold-change > 1 bit |
| `min_rmsd_to_phos` (phos MD) | < 6.0 Å | Phos MD actually visits the target state |
| `ref_apo_phos_rmsd_A` | > 5.0 Å | The two reference states are structurally distinct |

In the figures:
- **2D landscape**: Points cluster near the x-axis (apo MD) and near the
  y-axis (phos MD) — the two conditions occupy different quadrants.
- **Occupancy bar chart**: Phos-basin bar is much taller for phos MD than
  apo MD.
- **RMSF**: Phospho-site region (Y30, ±5 residues) shows increased
  flexibility in phos MD, indicating structural rearrangement.
- **Phosphate contacts**: R26 and R33 show high contact fractions (>0.30)
  in phos MD — these residues were designed to cradle the phosphate.

---

## Package structure

```
phosphoswitch-md/
├── src/phosphoswitch_md/
│   ├── trajectory.py   ← RMSD, RMSF, Rg, DSSP (MDAnalysis + GROMACS wrappers)
│   ├── basin.py        ← Dual-reference state assignment, occupancy metrics
│   ├── contacts.py     ← Phosphate-residue contacts, contact map clustering
│   ├── figures.py      ← All paper-quality plots (timeseries, landscapes, bars)
│   └── io_utils.py     ← XVG parsing, trajectory loading, candidate discovery
├── scripts/            ← Standalone Python scripts (call _cli/ internally)
├── config/
│   └── default.yaml    ← Default thresholds, selection strings, cutoffs
├── examples/
│   └── README.md       ← Step-by-step workflow
└── tests/
    ├── test_trajectory.py  ← Unit tests (MDAnalysis MemoryReader, no trajectories)
    └── test_basin.py       ← Unit tests for Kabsch RMSD, basin assignment
```

---

## Installation

```bash
git clone https://github.com/your-org/phosphoswitch-md
cd phosphoswitch-md
pip install -e .
```

Or with dev dependencies (testing):
```bash
pip install -e ".[dev]"
```

---

## Quick start

```bash
# 1. Postprocess raw trajectories (strip water, fit Cα)
postprocess --prod-dir prod_outputs/ --gmx /usr/local/gromacs/bin/gmx

# 2. Compute per-system analyses
analyze-trajectory \
    --prod-dir prod_outputs/ \
    --out-dir  md_analysis/ \
    --gmx gmx

# 3. Basin occupancy analysis (the main result)
basin-occupancy \
    --prod-dir prod_outputs/ \
    --af3-dir  af3_refs/ \
    --out-dir  md_analysis/basin_occupancy/

# 4. Generate all paper figures
make-figures \
    --analysis-dir md_analysis/ \
    --out-dir      md_figures/
```

---

## Python API

```python
import numpy as np
from phosphoswitch_md import (
    load_trajectory_universe,
    load_reference_coords,
    compute_dual_rmsd_series,
    compute_occupancy_metrics,
    compute_switch_metrics,
)

# Load a trajectory
u = load_trajectory_universe("prod_outputs/cand_12345_apo/rep1")

# Load AF3 reference coordinates
ref_apo  = load_reference_coords("af3_refs/cand_12345/apo.pdb")
ref_phos = load_reference_coords("af3_refs/cand_12345/phos.pdb")

# Compute dual-reference RMSD series
t, rmsd_to_apo, rmsd_to_phos = compute_dual_rmsd_series(
    u, ref_apo, ref_phos, n_sample=500
)

# Basin occupancy fractions
metrics = compute_occupancy_metrics(rmsd_to_apo, rmsd_to_phos,
                                    strict_A=4.0, loose_A=7.0)
print(f"Fraction in phos basin (loose): {metrics['frac_phos_loose']:.3f}")
```

---

## Running tests

```bash
pytest tests/ -v
```

All tests use synthetic coordinate data — no GROMACS trajectory files needed.

---

## Citation

If you use this package, please cite:

> Darby C. et al. (2026). Computational and NMR validation of designed
> phosphoswitch variants in LMNA. *In preparation*.

Analysis methodology follows Tesei et al. 2025 (bioRxiv 661537).

---

## License

MIT
