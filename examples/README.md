# MD Simulation Examples

## Movies (VMD-rendered, 600ns trajectories)

| File | Description |
|------|-------------|
| `movies/LMNA_WT.mp4` | LMNA Y45 construct, unphosphorylated. Predominantly helix conformation. |
| `movies/LMNA_PTR.mp4` | LMNA Y45 construct, phosphotyrosine (PTR). Shows hairpin adoption upon phosphorylation. |
| `movies/MAPRE1_WT.mp4` | MAPRE1 Y247 construct, unphosphorylated. |
| `movies/MAPRE1_PTR.mp4` | MAPRE1 Y247 construct, phosphotyrosine. Conformational change evident. |

These are 600ns GROMACS production runs at 300K in explicit TIP3P water, CHARMM36m force field.
The phosphotyrosine (PTR) CHARMM parameters are from Feng et al. 2021.

## Analysis figures

| File | Description |
|------|-------------|
| `figures/box_rmsd30_by_residue.png` | Per-residue Cα RMSD in the ±30aa window across all candidates |
| `figures/box_contact_by_residue.png` | Per-residue phosphate contact frequency, phos vs apo ensemble |

## Running the analysis

```bash
# Analyze a trajectory
psw-md-analyze-trajectory \
    --traj path/to/production.xtc \
    --top path/to/topol.gro \
    --ref1 path/to/stateA_phospho.pdb \
    --ref2 path/to/stateA_phospho_pulled.pdb \
    --out results/

# Basin occupancy
psw-md-basin-occupancy --manifest results/manifest.csv --out results/basin/

# Paper figures
psw-md-figures --results-dir results/ --out figures/
```
