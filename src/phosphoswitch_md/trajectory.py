"""
trajectory.py — Cα RMSD vs time, RMSF, Rg, and DSSP per-frame analyses.

All MDAnalysis-native implementations are preferred. GROMACS-wrapper functions
are also provided for reproducibility with upstream pipeline outputs.

Units
-----
- Positions: Å (MDAnalysis native)
- RMSD/RMSF/Rg returned in Å
- Time: nanoseconds
- DSSP: string codes per frame/residue
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .io_utils import (
    parse_xvg,
    parse_dssp_xpm,
    load_trajectory_universe,
    run_gmx,
)

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.rms import RMSD as MDA_RMSD, RMSF as MDA_RMSF
    HAS_MDA = True
except ImportError:
    HAS_MDA = False


# ---------------------------------------------------------------------------
# Atom selection helper
# ---------------------------------------------------------------------------

def select_protein_ca(universe) -> "mda.AtomGroup":
    """Select all protein Cα atoms, including phospho-amino-acid variants.

    MDAnalysis's ``protein`` keyword excludes HETATM records, so phospho
    residues (PTR, SEP, TPO, TP1, TP2) are explicitly included.
    """
    return universe.select_atoms(
        "(protein and name CA) or "
        "(resname PTR SEP TPO TP1 TP2 and name CA)"
    )


# ---------------------------------------------------------------------------
# MDAnalysis-native analyses
# ---------------------------------------------------------------------------

def compute_rmsd_mda(
    universe,
    ref_frame: int = 0,
    n_sample: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Cα RMSD vs a reference frame using MDAnalysis.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory Universe.
    ref_frame : int
        Frame index to use as the RMSD reference.
    n_sample : int, optional
        Subsample this many evenly-spaced frames. Uses all frames if None.

    Returns
    -------
    times_ns : np.ndarray
    rmsd_A : np.ndarray
        RMSD values in Å.
    """
    ca = select_protein_ca(universe)
    n_frames = len(universe.trajectory)
    indices = (
        np.linspace(0, n_frames - 1, n_sample, dtype=int)
        if n_sample and n_sample < n_frames
        else np.arange(n_frames)
    )

    # Store reference coordinates
    universe.trajectory[ref_frame]
    ref_coords = ca.positions.copy()

    times_ns = np.zeros(len(indices))
    rmsd_vals = np.zeros(len(indices))

    for i, fi in enumerate(indices):
        universe.trajectory[fi]
        coords = ca.positions
        # Centre both
        ref_c = ref_coords - ref_coords.mean(axis=0)
        cur_c = coords - coords.mean(axis=0)
        # Kabsch rotation
        H = cur_c.T @ ref_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        rotated = cur_c @ R
        rmsd_vals[i] = float(np.sqrt(((rotated - ref_c) ** 2).sum() / len(ref_c)))
        times_ns[i] = universe.trajectory.time / 1000.0

    return times_ns, rmsd_vals  # Å


def compute_rmsf_mda(
    universe,
    start_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-residue Cα RMSF over the last ``(1 - start_frac)`` of the trajectory.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    start_frac : float
        Skip this fraction of frames at the beginning (default 0.2 → use last 80%).

    Returns
    -------
    residue_ids : np.ndarray of int
        MDAnalysis residue IDs (1-based by default).
    rmsf_A : np.ndarray
        Per-residue RMSF in Å.
    """
    ca = select_protein_ca(universe)
    n_frames = len(universe.trajectory)
    start_frame = int(n_frames * start_frac)

    # Accumulate positions
    n_atoms = len(ca)
    coords_list = []
    for ts in universe.trajectory[start_frame:]:
        coords_list.append(ca.positions.copy())

    if not coords_list:
        return np.array([]), np.array([])

    coords = np.stack(coords_list)  # (T, N, 3)
    mean_pos = coords.mean(axis=0)  # (N, 3)
    rmsf_vals = np.sqrt(((coords - mean_pos) ** 2).sum(axis=2).mean(axis=0))  # (N,)

    residue_ids = np.array([r.resid for r in ca.residues])
    return residue_ids, rmsf_vals  # Å


def compute_rg_mda(
    universe,
    n_sample: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radius of gyration of the protein over time.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    n_sample : int, optional
        Subsample this many frames.

    Returns
    -------
    times_ns : np.ndarray
    rg_A : np.ndarray
        Radius of gyration in Å.
    """
    protein = universe.select_atoms("protein or resname PTR SEP TPO TP1 TP2")
    n_frames = len(universe.trajectory)
    indices = (
        np.linspace(0, n_frames - 1, n_sample, dtype=int)
        if n_sample and n_sample < n_frames
        else np.arange(n_frames)
    )

    times_ns = np.zeros(len(indices))
    rg_vals = np.zeros(len(indices))

    for i, fi in enumerate(indices):
        universe.trajectory[fi]
        pos = protein.positions
        com = pos.mean(axis=0)
        rg_vals[i] = float(np.sqrt(((pos - com) ** 2).sum(axis=1).mean()))
        times_ns[i] = universe.trajectory.time / 1000.0

    return times_ns, rg_vals  # Å


def compute_dssp_phi_psi(
    universe,
    n_sample: Optional[int] = 100,
) -> np.ndarray:
    """Estimate per-residue secondary structure from phi/psi dihedral angles.

    This is a geometry-based proxy for DSSP (no external binary needed).
    Helix region: -100 < phi < -30, -80 < psi < -5 (classic alpha-helix basin)
    Sheet region: phi < -50, (psi > 100 or psi < -150)

    Parameters
    ----------
    universe : MDAnalysis.Universe
    n_sample : int, optional
        Number of frames to sample. Uses all if None.

    Returns
    -------
    ss_fracs : np.ndarray, shape (n_residues, 3)
        Columns: [helix_fraction, sheet_fraction, coil_fraction]
        Indexed in residue order. Terminal residues (lacking neighbours for
        phi or psi) will have all-zero rows.
    """
    residues = universe.select_atoms("protein").residues
    n_res = len(residues)
    n_frames = len(universe.trajectory)
    indices = (
        np.linspace(0, n_frames - 1, n_sample, dtype=int)
        if n_sample and n_sample < n_frames
        else np.arange(n_frames)
    )

    helix = np.zeros(n_res)
    sheet = np.zeros(n_res)
    total = np.zeros(n_res)

    # Precompute phi/psi atom selections per residue
    rama_sels = []
    for r in residues:
        rn = r.resid
        try:
            phi_atoms = universe.select_atoms(
                f"(resid {rn - 1} and name C) or "
                f"(resid {rn} and (name N or name CA or name C))"
            )
            psi_atoms = universe.select_atoms(
                f"(resid {rn} and (name N or name CA or name C)) or "
                f"(resid {rn + 1} and name N)"
            )
            if len(phi_atoms) == 4 and len(psi_atoms) == 4:
                rama_sels.append((phi_atoms, psi_atoms))
            else:
                rama_sels.append(None)
        except Exception:
            rama_sels.append(None)

    for fi in indices:
        universe.trajectory[fi]
        for ri, sels in enumerate(rama_sels):
            if sels is None:
                continue
            phi_atoms, psi_atoms = sels
            try:
                phi = phi_atoms.dihedral.value()
                psi = psi_atoms.dihedral.value()
            except Exception:
                continue
            total[ri] += 1
            if -100 < phi < -30 and -80 < psi < -5:
                helix[ri] += 1
            elif phi < -50 and (psi > 100 or psi < -150):
                sheet[ri] += 1

    helix_f = np.where(total > 0, helix / total, 0.0)
    sheet_f = np.where(total > 0, sheet / total, 0.0)
    coil_f = 1.0 - helix_f - sheet_f

    return np.column_stack([helix_f, sheet_f, coil_f])


# ---------------------------------------------------------------------------
# GROMACS wrapper functions
# ---------------------------------------------------------------------------

def run_gmx_rms(
    rep_dir: Union[str, Path],
    gmx: str = "gmx",
    ana_dir: Optional[Union[str, Path]] = None,
    traj: str = "prod_fit.xtc",
    tpr: str = "prod.tpr",
    ca_group: str = "3",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Cα RMSD vs first frame using ``gmx rms``.

    Parameters
    ----------
    rep_dir : path
        Replicate directory (must contain *tpr* and *traj*).
    gmx : str
        Path or name of the ``gmx`` binary.
    ana_dir : path, optional
        Where to write the ``rmsd.xvg`` output. Defaults to ``rep_dir/analysis/``.
    traj, tpr : str
        Filenames relative to *rep_dir*.
    ca_group : str
        Group index for Cα (typically "3").

    Returns
    -------
    times_ns, rmsd_nm : (np.ndarray, np.ndarray)
        Values in ns and nm as output by GROMACS.
    """
    rep_dir = Path(rep_dir)
    if ana_dir is None:
        ana_dir = rep_dir / "analysis"
    ana_dir = Path(ana_dir)
    ana_dir.mkdir(parents=True, exist_ok=True)

    out = ana_dir / "rmsd.xvg"
    if not out.is_file():
        run_gmx(
            [gmx, "rms", "-s", tpr, "-f", traj, "-o", str(out), "-fit", "rot+trans"],
            cwd=rep_dir,
            stdin=f"{ca_group}\n{ca_group}\n",
        )
    return parse_xvg(out)


def run_gmx_rg(
    rep_dir: Union[str, Path],
    gmx: str = "gmx",
    ana_dir: Optional[Union[str, Path]] = None,
    traj: str = "prod_fit.xtc",
    tpr: str = "prod.tpr",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radius of gyration using ``gmx gyrate``.

    Returns (times_ns, rg_nm).
    """
    rep_dir = Path(rep_dir)
    if ana_dir is None:
        ana_dir = rep_dir / "analysis"
    ana_dir = Path(ana_dir)
    ana_dir.mkdir(parents=True, exist_ok=True)

    out = ana_dir / "rg.xvg"
    if not out.is_file():
        run_gmx(
            [gmx, "gyrate", "-s", tpr, "-f", traj, "-o", str(out)],
            cwd=rep_dir,
            stdin="1\n",  # 1 = Protein
        )
    return parse_xvg(out)


def run_gmx_rmsf(
    rep_dir: Union[str, Path],
    gmx: str = "gmx",
    ana_dir: Optional[Union[str, Path]] = None,
    traj: str = "prod_fit.xtc",
    tpr: str = "prod.tpr",
    start_ps: int = 120_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-residue RMSF using ``gmx rmsf``.

    Parameters
    ----------
    start_ps : int
        Start time in picoseconds (default 120,000 ps = 120 ns = last 80% of 600 ns).

    Returns
    -------
    residue_ids : np.ndarray of int
    rmsf_nm : np.ndarray
        Per-residue RMSF in nm (GROMACS native unit).
    """
    rep_dir = Path(rep_dir)
    if ana_dir is None:
        ana_dir = rep_dir / "analysis"
    ana_dir = Path(ana_dir)
    ana_dir.mkdir(parents=True, exist_ok=True)

    out = ana_dir / "rmsf.xvg"
    if not out.is_file():
        run_gmx(
            [gmx, "rmsf", "-s", tpr, "-f", traj, "-o", str(out),
             "-res", "-b", str(start_ps)],
            cwd=rep_dir,
            stdin="1\n",  # Protein
        )

    residues, values = [], []
    with open(out) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    residues.append(int(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    pass

    return np.array(residues), np.array(values)


def run_gmx_dssp(
    rep_dir: Union[str, Path],
    gmx: str = "gmx",
    ana_dir: Optional[Union[str, Path]] = None,
    traj: str = "prod_fit.xtc",
    tpr: str = "prod.tpr",
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """Assign secondary structure using ``gmx dssp`` (or ``do_dssp``).

    Returns parsed XPM matrix and code map, or (None, None) if DSSP is
    unavailable in the installed GROMACS build.
    """
    rep_dir = Path(rep_dir)
    if ana_dir is None:
        ana_dir = rep_dir / "analysis"
    ana_dir = Path(ana_dir)
    ana_dir.mkdir(parents=True, exist_ok=True)

    out_xpm = ana_dir / "dssp.xpm"
    if not out_xpm.is_file():
        for tool in ("dssp", "do_dssp"):
            try:
                run_gmx(
                    [gmx, tool, "-s", tpr, "-f", traj, "-o", str(out_xpm)],
                    cwd=rep_dir,
                    stdin="1\n",
                    check=True,
                )
                break
            except RuntimeError:
                continue
        else:
            return None, None  # DSSP not available

    return parse_dssp_xpm(out_xpm)


# ---------------------------------------------------------------------------
# Sidechain SASA (via gmx sasa)
# ---------------------------------------------------------------------------

def run_gmx_sasa(
    rep_dir: Union[str, Path],
    gmx: str = "gmx",
    ana_dir: Optional[Union[str, Path]] = None,
    phos_resnum: int = 30,
    traj: str = "prod_fit.xtc",
    tpr: str = "prod.tpr",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SASA of the phospho-tyrosine sidechain using ``gmx sasa``.

    Returns (times_ns, sasa_nm2).
    """
    rep_dir = Path(rep_dir)
    if ana_dir is None:
        ana_dir = rep_dir / "analysis"
    ana_dir = Path(ana_dir)
    ana_dir.mkdir(parents=True, exist_ok=True)

    out = ana_dir / "sasa.xvg"
    if not out.is_file():
        idx_file = ana_dir / "phos.ndx"
        # Build index for sidechain atoms of the phospho residue
        run_gmx(
            [gmx, "make_ndx", "-f", tpr, "-o", str(idx_file)],
            cwd=rep_dir,
            stdin=(
                f"r {phos_resnum} & ! a N CA C O H HA\n"
                "name 19 phos_sidechain\n"
                "q\n"
            ),
        )
        run_gmx(
            [gmx, "sasa", "-s", tpr, "-f", traj, "-o", str(out),
             "-n", str(idx_file)],
            cwd=rep_dir,
            stdin="phos_sidechain\n",
        )

    return parse_xvg(out)


# ---------------------------------------------------------------------------
# Aggregate one replicate's analyses to CSV
# ---------------------------------------------------------------------------

def run_all_analyses_one_rep(
    rep_dir: Union[str, Path],
    gmx: str = "gmx",
    phos_resnum: int = 30,
    rmsf_start_ps: int = 120_000,
    use_mda_fallback: bool = True,
) -> dict:
    """Run RMSD, Rg, RMSF, SASA (and optionally DSSP) for one replicate.

    Returns a dictionary of metric names → scalar summary values. The full
    time-series xvg files are written to ``rep_dir/analysis/``.
    """
    rep_dir = Path(rep_dir)
    ana_dir = rep_dir / "analysis"
    ana_dir.mkdir(exist_ok=True)

    results = {}

    # RMSD (gmx)
    try:
        t, rmsd = run_gmx_rms(rep_dir, gmx, ana_dir)
        results["rmsd_final_nm"] = float(rmsd[-100:].mean()) if len(rmsd) >= 100 else float(rmsd.mean())
    except Exception as e:
        results["rmsd_final_nm"] = None
        results["rmsd_error"] = str(e)

    # Rg (gmx)
    try:
        t, rg = run_gmx_rg(rep_dir, gmx, ana_dir)
        results["rg_final_nm"] = float(rg[-100:].mean()) if len(rg) >= 100 else float(rg.mean())
    except Exception as e:
        results["rg_final_nm"] = None

    # RMSF (gmx)
    try:
        res, rmsf = run_gmx_rmsf(rep_dir, gmx, ana_dir, rmsf_start_ps)
        results["rmsf_mean_nm"] = float(rmsf.mean()) if len(rmsf) > 0 else None
    except Exception as e:
        results["rmsf_mean_nm"] = None

    # SASA (gmx, phos sims only)
    try:
        t, sasa = run_gmx_sasa(rep_dir, gmx, ana_dir, phos_resnum)
        results["sasa_phos_mean_nm2"] = float(sasa.mean()) if len(sasa) > 0 else None
    except Exception as e:
        results["sasa_phos_mean_nm2"] = None

    return results
