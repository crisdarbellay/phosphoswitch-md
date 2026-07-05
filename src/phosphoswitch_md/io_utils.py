"""
io_utils.py — GROMACS output reading, replicate discovery, and results aggregation.

Handles:
  - Parsing GROMACS .xvg files (time-series) and .xpm matrices (DSSP)
  - Loading MDAnalysis Universes from replicate directories
  - Finding candidate directories with the expected naming convention
  - Locating AF3 reference PDB files
  - Aggregating multi-replicate time-series to mean ± SEM
  - Reading analysis result CSVs back into NumPy arrays
  - Running gmx subprocess commands
"""

import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False


# ---------------------------------------------------------------------------
# GROMACS .xvg parsing
# ---------------------------------------------------------------------------

def parse_xvg(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a two-column GROMACS .xvg file.

    Returns
    -------
    times_ns : np.ndarray
        Time axis in nanoseconds (converted from ps if necessary).
    values : np.ndarray
        Second column values in the native GROMACS unit (nm, nm², etc.).
    """
    times, values = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                v = float(parts[1])
            except ValueError:
                continue
            times.append(t)
            values.append(v)

    times_arr = np.array(times)
    values_arr = np.array(values)

    # GROMACS outputs time in ps; convert to ns if needed
    if len(times_arr) > 0 and times_arr[-1] > 1000:
        times_arr = times_arr / 1000.0

    return times_arr, values_arr


def parse_xvg_multi(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a multi-column GROMACS .xvg file.

    Returns
    -------
    times_ns : np.ndarray, shape (T,)
    data : np.ndarray, shape (T, ncols-1)
    """
    times, rows = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                vs = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            times.append(t)
            rows.append(vs)

    times_arr = np.array(times)
    data_arr = np.array(rows) if rows else np.zeros((0, 0))

    if len(times_arr) > 0 and times_arr[-1] > 1000:
        times_arr = times_arr / 1000.0

    return times_arr, data_arr


def parse_dssp_xpm(xpm_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """Parse a GROMACS dssp .xpm matrix into a NumPy array of SS codes.

    Returns
    -------
    arr : np.ndarray or None
        Object array of shape (n_frames, n_residues) with DSSP code strings.
    code_map : dict or None
        Mapping from single-character code → label string.
    """
    code_map: dict = {}
    rows: List[List[str]] = []
    width = height = n_colors = 0

    with open(xpm_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(("/*", "static char", "}")):
                continue
            if not line.startswith('"'):
                continue
            if not width:
                # First data line: "WIDTH HEIGHT NCOLORS CHARS_PER_PIXEL"
                parts = line.strip('",').split()
                try:
                    width, height, n_colors = int(parts[0]), int(parts[1]), int(parts[2])
                except (ValueError, IndexError):
                    pass
                continue
            if len(code_map) < n_colors:
                # Color definition lines: "X c #RRGGBB " /* "label" */
                inner = line.strip('"')
                char = inner[0] if inner else None
                label_start = inner.find('"', 1)
                label_end = inner.find('"', label_start + 1)
                if char and label_start != -1 and label_end != -1:
                    label = inner[label_start + 1: label_end]
                    code_map[char] = label
                continue
            # Data row
            row_str = line.strip('",')
            if len(row_str) == width:
                rows.append([code_map.get(c, "?") for c in row_str])

    if not rows:
        return None, None

    return np.array(rows, dtype=object), code_map


# ---------------------------------------------------------------------------
# Aggregated CSV I/O
# ---------------------------------------------------------------------------

def read_aggregated_csv(path: Union[str, Path]) -> Optional[Dict[str, np.ndarray]]:
    """Read an aggregated result CSV (e.g. rmsd_mean_sem.csv) into a dict of arrays.

    Returns None if the file does not exist or is empty.
    """
    p = Path(path)
    if not p.is_file():
        return None

    rows = []
    with open(p) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: float(v) if v else np.nan for k, v in row.items()})

    if not rows:
        return None

    return {col: np.array([r[col] for r in rows]) for col in rows[0]}


def aggregate_replicates(
    time_series_list: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine multiple (time, value) time-series into mean ± SEM on a common grid.

    All series are interpolated onto the first replicate's time grid.

    Parameters
    ----------
    time_series_list : list of (time_ns, values) tuples

    Returns
    -------
    ref_t : np.ndarray
    mean : np.ndarray
    sem : np.ndarray
    """
    if not time_series_list:
        return np.array([]), np.array([]), np.array([])

    ref_t = time_series_list[0][0]
    aligned = []
    for t, v in time_series_list:
        if len(v) == len(ref_t):
            aligned.append(v)
        else:
            aligned.append(np.interp(ref_t, t, v))

    arr = np.array(aligned)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem = np.zeros_like(mean)

    return ref_t, mean, sem


def write_mean_sem_csv(
    path: Union[str, Path],
    ref_t: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    col_names: Tuple[str, str, str] = ("time_ns", "mean", "sem"),
) -> None:
    """Write a three-column mean ± SEM CSV."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(col_names))
        for t, m, s in zip(ref_t, mean, sem):
            w.writerow([f"{t:.3f}", f"{m:.6f}", f"{s:.6f}"])


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def find_candidates(prod_dir: Union[str, Path]) -> List[str]:
    """Find candidate base names that have both ``<cand>_apo`` and ``<cand>_phos`` subdirs.

    Expected layout::

        prod_dir/
          cand_XXX_<tag>_apo/   ← apo simulation
          cand_XXX_<tag>_phos/  ← phos simulation

    Returns a sorted list of base names (everything before the trailing ``_apo``/``_phos``).
    """
    prod_dir = Path(prod_dir)
    candidates: dict = defaultdict(set)
    for d in prod_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.endswith("_apo"):
            candidates[d.name[:-4]].add("apo")
        elif d.name.endswith("_phos"):
            candidates[d.name[:-5]].add("phos")

    return sorted(
        c for c, conds in candidates.items() if "apo" in conds and "phos" in conds
    )


def get_rep_dirs(system_dir: Union[str, Path]) -> List[Path]:
    """Return sorted list of ``rep*/`` subdirectories inside *system_dir*."""
    system_dir = Path(system_dir)
    return sorted(
        d for d in system_dir.iterdir() if d.is_dir() and d.name.startswith("rep")
    )


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectory_universe(rep_dir: Union[str, Path]):
    """Load an MDAnalysis Universe for one replicate directory.

    Prefers ``prod_fit.xtc + prod.pdb`` (stripped, fitted).
    Falls back to ``prod.xtc + prod.tpr`` (full trajectory).

    Returns ``None`` if topology or trajectory file is missing.
    """
    if not HAS_MDA:
        raise ImportError("MDAnalysis is required. Install with: pip install MDAnalysis")

    rep_dir = Path(rep_dir)
    fit_xtc = rep_dir / "prod_fit.xtc"
    pdb = rep_dir / "prod.pdb"
    tpr = rep_dir / "prod.tpr"
    full_xtc = rep_dir / "prod.xtc"

    if fit_xtc.exists():
        topo = pdb if pdb.exists() else tpr
        traj = fit_xtc
    else:
        topo = tpr
        traj = full_xtc

    if not topo.exists() or not traj.exists():
        return None

    return mda.Universe(str(topo), str(traj))


# ---------------------------------------------------------------------------
# AF3 reference locator
# ---------------------------------------------------------------------------

def find_af3_references(
    af3_dir: Union[str, Path], candidate_base: str
) -> Tuple[Optional[Path], Optional[Path]]:
    """Locate the AF3-predicted apo and phos reference PDB files.

    Tries several common directory layouts::

        <af3_dir>/<candidate_base>/apo.pdb  + phos.pdb
        <af3_dir>/<candidate_base>/wt.pdb   + phos.pdb
        <af3_dir>/<candidate_base>_apo/start.pdb + <cand>_phos/start.pdb
        <af3_dir>/<candidate_base>_apo.pdb  + <cand>_phos.pdb
        <af3_dir>/<candidate_base>/model_apo.pdb + model_phos.pdb

    Returns ``(None, None)`` if neither layout matches.
    """
    af3_dir = Path(af3_dir)
    b = candidate_base
    layouts = [
        (af3_dir / b / "apo.pdb", af3_dir / b / "phos.pdb"),
        (af3_dir / b / "wt.pdb", af3_dir / b / "phos.pdb"),
        (af3_dir / f"{b}_apo" / "start.pdb", af3_dir / f"{b}_phos" / "start.pdb"),
        (af3_dir / f"{b}_apo.pdb", af3_dir / f"{b}_phos.pdb"),
        (af3_dir / b / "model_apo.pdb", af3_dir / b / "model_phos.pdb"),
    ]
    for apo, phos in layouts:
        if apo.exists() and phos.exists():
            return apo, phos

    return None, None


# ---------------------------------------------------------------------------
# GROMACS subprocess wrapper
# ---------------------------------------------------------------------------

def run_gmx(
    cmd: List[str],
    cwd: Optional[Union[str, Path]] = None,
    stdin: Optional[str] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a GROMACS command, raising ``RuntimeError`` on non-zero exit.

    Parameters
    ----------
    cmd : list of str
        e.g. ``["gmx", "rms", "-s", "prod.tpr", ...]``
    cwd : path, optional
        Working directory for the subprocess.
    stdin : str, optional
        Text piped to stdin (e.g. group-selection prompts).
    check : bool
        If True (default), raise on failure.
    """
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=stdin,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        cmd_str = " ".join(str(c) for c in cmd)
        raise RuntimeError(
            f"gmx failed (rc={result.returncode}): {cmd_str}\n"
            f"STDOUT: {result.stdout[-500:]}\n"
            f"STDERR: {result.stderr[-500:]}"
        )
    return result
