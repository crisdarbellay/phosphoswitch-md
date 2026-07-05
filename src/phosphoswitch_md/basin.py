"""
basin.py — Dual-reference state assignment and basin occupancy analysis.

Biology
-------
Each LMNA Y45 designed variant is characterised by two AF3-predicted reference
structures:

  * **Apo reference** — the straight-helix conformation (unphosphorylated).
  * **Phos reference** — the bent/hairpin conformation (phosphorylated Y30/PTR).

For every frame of every replicate MD trajectory, we compute the Kabsch-aligned
Cα RMSD to both references and assign the frame to the *closer* basin:

    basin(frame) = "apo"  if RMSD_to_apo  < RMSD_to_phos
                 = "phos" if RMSD_to_phos <= RMSD_to_apo

A **successful phosphoswitch** shows:
  * Apo MD:  most frames in the apo basin (RMSD_to_apo < strict threshold).
  * Phos MD: most frames in the phos basin (RMSD_to_phos < strict threshold).
  * Switch strength ≡ |frac_apo_basin(apo MD) − frac_apo_basin(phos MD)| > 0.3.
  * Phos-basin enrichment = frac_phos(phos MD) / frac_phos(apo MD) >> 1.

Algorithms
----------
All RMSD calculations use the Kabsch algorithm (optimal rotation) on the Cα
atoms, matching Tesei et al. 2025 and the upstream 12_/14_ scripts.

Units
-----
All RMSD values returned in Ångströms.  Thresholds expressed in Å.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .io_utils import load_trajectory_universe

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False


# ---------------------------------------------------------------------------
# Kabsch RMSD (core computation)
# ---------------------------------------------------------------------------

def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Optimal Kabsch-aligned RMSD between two sets of 3-D coordinates.

    Parameters
    ----------
    coords_a, coords_b : np.ndarray, shape (N, 3)
        Cα coordinate arrays in Å.  Must have the same length N.

    Returns
    -------
    float
        RMSD in Å after optimal rotation of *coords_a* onto *coords_b*.
    """
    A = coords_a - coords_a.mean(axis=0)
    B = coords_b - coords_b.mean(axis=0)
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Correct for reflection (determinant should be +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    A_rot = A @ R
    return float(np.sqrt(((A_rot - B) ** 2).sum() / len(A)))


# ---------------------------------------------------------------------------
# Atom selection
# ---------------------------------------------------------------------------

def select_protein_ca(universe) -> "mda.AtomGroup":
    """Select Cα atoms from protein and phospho-amino-acid residues.

    Includes PTR (phospho-Tyr), SEP (phospho-Ser), TPO (phospho-Thr),
    TP1, TP2 because AF3 phos.pdb stores these as HETATM, which the
    MDAnalysis ``protein`` keyword excludes.
    """
    return universe.select_atoms(
        "(protein and name CA) or "
        "(resname PTR SEP TPO TP1 TP2 and name CA)"
    )


# ---------------------------------------------------------------------------
# Reference coordinate loading
# ---------------------------------------------------------------------------

def load_reference_coords(pdb_path: Union[str, Path]) -> np.ndarray:
    """Load Cα coordinates from an AF3 reference PDB file.

    Parameters
    ----------
    pdb_path : path
        Path to a PDB file (typically the AF3 apo or phos prediction).

    Returns
    -------
    coords : np.ndarray, shape (N_ca, 3)
        Cα positions in Å.
    """
    if not HAS_MDA:
        raise ImportError("MDAnalysis is required. Install with: pip install MDAnalysis")

    u = mda.Universe(str(pdb_path))
    ca = select_protein_ca(u)
    if len(ca) == 0:
        raise ValueError(f"No Cα atoms found in {pdb_path}")
    return ca.positions.copy()


# ---------------------------------------------------------------------------
# Dual-reference RMSD series
# ---------------------------------------------------------------------------

def compute_dual_rmsd_series(
    universe,
    ref_apo_coords: np.ndarray,
    ref_phos_coords: np.ndarray,
    n_sample: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame Cα RMSD to *both* reference structures.

    For each sampled frame, computes Kabsch-aligned RMSD to the apo reference
    and the phos reference independently.  This is the core measurement used
    in the 2D landscape and basin occupancy analyses.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory.
    ref_apo_coords : np.ndarray, shape (N, 3)
        Cα coordinates of the AF3 apo (straight-helix) reference in Å.
    ref_phos_coords : np.ndarray, shape (N, 3)
        Cα coordinates of the AF3 phos (hairpin) reference in Å.
    n_sample : int, optional
        Subsample evenly to this many frames.  Default: all frames.

    Returns
    -------
    times_ns : np.ndarray, shape (T,)
    rmsd_to_apo_A : np.ndarray, shape (T,)
        RMSD to the apo reference in Å.
    rmsd_to_phos_A : np.ndarray, shape (T,)
        RMSD to the phos reference in Å.
    """
    ca = select_protein_ca(universe)
    n_res_traj = len(ca)

    # Align reference arrays to the smallest common residue count
    n_min = min(n_res_traj, len(ref_apo_coords), len(ref_phos_coords))
    ref_apo = ref_apo_coords[:n_min]
    ref_phos = ref_phos_coords[:n_min]

    n_frames = len(universe.trajectory)
    indices = (
        np.linspace(0, n_frames - 1, n_sample, dtype=int)
        if n_sample and n_sample < n_frames
        else np.arange(n_frames)
    )

    times_ns = np.zeros(len(indices))
    rmsd_apo = np.zeros(len(indices))
    rmsd_phos = np.zeros(len(indices))

    for i, fi in enumerate(indices):
        universe.trajectory[fi]
        coords = ca.positions[:n_min].copy()
        rmsd_apo[i] = kabsch_rmsd(coords, ref_apo)
        rmsd_phos[i] = kabsch_rmsd(coords, ref_phos)
        times_ns[i] = universe.trajectory.time / 1000.0

    return times_ns, rmsd_apo, rmsd_phos


def compute_dual_rmsd_from_dir(
    rep_dir: Union[str, Path],
    ref_apo_path: Union[str, Path],
    ref_phos_path: Union[str, Path],
    n_sample: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Convenience wrapper: load trajectory from *rep_dir*, run dual-RMSD.

    Returns
    -------
    dict with keys ``time_ns``, ``rmsd_to_apo``, ``rmsd_to_phos``, or None
    if the trajectory cannot be loaded.
    """
    universe = load_trajectory_universe(rep_dir)
    if universe is None:
        return None

    ref_apo = load_reference_coords(ref_apo_path)
    ref_phos = load_reference_coords(ref_phos_path)

    t, ra, rp = compute_dual_rmsd_series(universe, ref_apo, ref_phos, n_sample)
    return {"time_ns": t, "rmsd_to_apo": ra, "rmsd_to_phos": rp}


# ---------------------------------------------------------------------------
# Basin assignment
# ---------------------------------------------------------------------------

def assign_basin_per_frame(
    rmsd_to_apo: np.ndarray,
    rmsd_to_phos: np.ndarray,
) -> np.ndarray:
    """Assign each frame to the closer reference basin.

    Parameters
    ----------
    rmsd_to_apo, rmsd_to_phos : np.ndarray, shape (T,)
        Per-frame RMSD values in Å.

    Returns
    -------
    basin : np.ndarray of str, shape (T,)
        Each element is ``"apo"`` or ``"phos"``.
    """
    return np.where(rmsd_to_apo <= rmsd_to_phos, "apo", "phos")


# ---------------------------------------------------------------------------
# Occupancy metrics (from 14_basin_occupancy_analysis.py)
# ---------------------------------------------------------------------------

def compute_occupancy_metrics(
    rmsd_to_apo: np.ndarray,
    rmsd_to_phos: np.ndarray,
    strict_A: float = 4.0,
    loose_A: float = 7.0,
) -> Dict[str, float]:
    """Compute basin occupancy statistics from dual-RMSD arrays.

    The *strict* threshold (4 Å default) defines tight basin membership:
    a frame must be very close to the reference to count.  The *loose*
    threshold (7 Å default) captures nearby visits including intermediate
    conformations.

    Parameters
    ----------
    rmsd_to_apo, rmsd_to_phos : np.ndarray
        Per-frame RMSD in Å.
    strict_A, loose_A : float
        Basin-membership cutoffs in Å.

    Returns
    -------
    dict with keys:

    * ``n_frames`` — total frames analysed
    * ``min_rmsd_to_apo``, ``min_rmsd_to_phos`` — global minima
    * ``median_rmsd_to_apo``, ``median_rmsd_to_phos`` — medians
    * ``mean_rmsd_to_apo``, ``mean_rmsd_to_phos`` — means
    * ``frac_apo_strict`` — fraction of frames with RMSD_to_apo < strict_A
    * ``frac_apo_loose``  — fraction of frames with RMSD_to_apo < loose_A
    * ``frac_phos_strict`` — fraction of frames with RMSD_to_phos < strict_A
    * ``frac_phos_loose``  — fraction of frames with RMSD_to_phos < loose_A
    * ``frac_neither_loose`` — fraction with both RMSD values > loose_A
    * ``frac_both_loose`` — fraction within loose_A of both references
    """
    n = len(rmsd_to_apo)
    return {
        "n_frames": n,
        "min_rmsd_to_apo": float(np.min(rmsd_to_apo)),
        "min_rmsd_to_phos": float(np.min(rmsd_to_phos)),
        "median_rmsd_to_apo": float(np.median(rmsd_to_apo)),
        "median_rmsd_to_phos": float(np.median(rmsd_to_phos)),
        "mean_rmsd_to_apo": float(np.mean(rmsd_to_apo)),
        "mean_rmsd_to_phos": float(np.mean(rmsd_to_phos)),
        "frac_apo_strict": float((rmsd_to_apo < strict_A).mean()),
        "frac_apo_loose": float((rmsd_to_apo < loose_A).mean()),
        "frac_phos_strict": float((rmsd_to_phos < strict_A).mean()),
        "frac_phos_loose": float((rmsd_to_phos < loose_A).mean()),
        "frac_neither_loose": float(
            ((rmsd_to_apo >= loose_A) & (rmsd_to_phos >= loose_A)).mean()
        ),
        "frac_both_loose": float(
            ((rmsd_to_apo < loose_A) & (rmsd_to_phos < loose_A)).mean()
        ),
    }


def find_hitting_time(
    time_ns: np.ndarray,
    rmsd_to_phos: np.ndarray,
    threshold_A: float = 4.0,
) -> float:
    """Return the first time (ns) when RMSD_to_phos drops below *threshold_A*.

    Returns ``float("nan")`` if the threshold is never reached.  This is the
    *first-passage time* to the phos basin, relevant for kinetics.
    """
    hits = np.where(rmsd_to_phos < threshold_A)[0]
    if len(hits) == 0:
        return float("nan")
    return float(time_ns[hits[0]])


# ---------------------------------------------------------------------------
# Switch quality metrics (from 14_ enrichment calculations)
# ---------------------------------------------------------------------------

def compute_switch_metrics(
    apo_metrics: Dict[str, float],
    phos_metrics: Dict[str, float],
    eps: float = 1e-4,
) -> Dict[str, float]:
    """Compute phosphoswitch quality from occupancy metrics of apo and phos MD.

    The core metric is the *phos-basin enrichment ratio*:

        enrich_phos_loose = frac_phos(phos MD) / frac_phos(apo MD)

    A ratio > 1 means phosphorylation biases the trajectory toward the phos
    basin.  The log2 form is more readable: +1 = 2× enrichment.

    A strong switch shows:
      * enrich_phos_loose >> 1 (phos MD spends far more time near phos state)
      * enrich_apo_loose  << 1 or ≈ 1 (both MDs access the apo basin)
      * switch_strength > 0.3 (Δ fraction in apo basin between conditions)

    Parameters
    ----------
    apo_metrics, phos_metrics : dict
        Output of :func:`compute_occupancy_metrics` for the apo and phos MD
        ensembles respectively.

    Returns
    -------
    dict with keys:
      ``enrich_phos_strict``, ``enrich_phos_loose``,
      ``log2_phos_enrich_strict``, ``log2_phos_enrich_loose``,
      ``enrich_apo_strict``, ``enrich_apo_loose``,
      ``log2_apo_enrich_strict``, ``log2_apo_enrich_loose``,
      ``switch_strength`` (|Δ frac_apo_basin| between conditions)
    """
    def ratio(a_key, b_key):
        return (phos_metrics[b_key] + eps) / (apo_metrics[a_key] + eps)

    ep_strict = ratio("frac_phos_strict", "frac_phos_strict")
    ep_loose = ratio("frac_phos_loose", "frac_phos_loose")
    ea_strict = ratio("frac_apo_strict", "frac_apo_strict")
    ea_loose = ratio("frac_apo_loose", "frac_apo_loose")

    switch_strength = abs(apo_metrics["frac_apo_loose"] - phos_metrics["frac_apo_loose"])

    return {
        "enrich_phos_strict": round(ep_strict, 3),
        "enrich_phos_loose": round(ep_loose, 3),
        "log2_phos_enrich_strict": round(float(np.log2(ep_strict)), 3),
        "log2_phos_enrich_loose": round(float(np.log2(ep_loose)), 3),
        "enrich_apo_strict": round(ea_strict, 3),
        "enrich_apo_loose": round(ea_loose, 3),
        "log2_apo_enrich_strict": round(float(np.log2(ea_strict)), 3),
        "log2_apo_enrich_loose": round(float(np.log2(ea_loose)), 3),
        "switch_strength": round(switch_strength, 3),
    }


# ---------------------------------------------------------------------------
# Full candidate analysis
# ---------------------------------------------------------------------------

def analyze_candidate_basin_occupancy(
    candidate: str,
    prod_dir: Union[str, Path],
    ref_apo_path: Union[str, Path],
    ref_phos_path: Union[str, Path],
    n_sample: int = 1000,
    strict_A: float = 4.0,
    loose_A: float = 7.0,
) -> Dict:
    """Run complete dual-RMSD basin analysis for one candidate.

    Iterates over all replicates in ``<prod_dir>/<candidate>_apo/`` and
    ``<prod_dir>/<candidate>_phos/``, computes dual-RMSD series and
    occupancy metrics, then aggregates across replicates.

    Returns a nested dict::

        {
          "apo":  {"reps": [...], "aggregated": {...occupancy_metrics...}},
          "phos": {"reps": [...], "aggregated": {...occupancy_metrics...}},
          "switch_metrics": {...},
          "ref_rmsd_A": float,   # distance between the two reference structures
        }
    """
    from .io_utils import get_rep_dirs

    prod_dir = Path(prod_dir)
    ref_apo = load_reference_coords(ref_apo_path)
    ref_phos = load_reference_coords(ref_phos_path)

    n_min = min(len(ref_apo), len(ref_phos))
    ref_apo = ref_apo[:n_min]
    ref_phos = ref_phos[:n_min]

    ref_rmsd = kabsch_rmsd(ref_apo, ref_phos)

    result: Dict = {
        "ref_rmsd_A": round(ref_rmsd, 3),
        "apo": {"reps": []},
        "phos": {"reps": []},
    }

    for cond in ("apo", "phos"):
        sys_dir = prod_dir / f"{candidate}_{cond}"
        if not sys_dir.exists():
            continue
        all_rmsd_apo = []
        all_rmsd_phos = []
        all_times = []
        for rep_dir in get_rep_dirs(sys_dir):
            universe = load_trajectory_universe(rep_dir)
            if universe is None:
                continue
            t, ra, rp = compute_dual_rmsd_series(universe, ref_apo, ref_phos, n_sample)
            rep_data = {
                "rep": rep_dir.name,
                "time_ns": t,
                "rmsd_to_apo": ra,
                "rmsd_to_phos": rp,
                "metrics": compute_occupancy_metrics(ra, rp, strict_A, loose_A),
                "hitting_time_strict_ns": find_hitting_time(t, rp, strict_A),
                "hitting_time_loose_ns": find_hitting_time(t, rp, loose_A),
            }
            result[cond]["reps"].append(rep_data)
            all_rmsd_apo.append(ra)
            all_rmsd_phos.append(rp)
            all_times.append(t)

        if all_rmsd_apo:
            pooled_apo = np.concatenate(all_rmsd_apo)
            pooled_phos = np.concatenate(all_rmsd_phos)
            result[cond]["aggregated"] = compute_occupancy_metrics(
                pooled_apo, pooled_phos, strict_A, loose_A
            )
        else:
            result[cond]["aggregated"] = {}

    agg_apo = result["apo"].get("aggregated", {})
    agg_phos = result["phos"].get("aggregated", {})
    if agg_apo and agg_phos:
        result["switch_metrics"] = compute_switch_metrics(agg_apo, agg_phos)
    else:
        result["switch_metrics"] = {}

    return result
