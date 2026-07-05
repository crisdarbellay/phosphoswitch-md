"""
contacts.py — Phosphate-residue contacts and Cα-Cα contact map analysis.

Two complementary analyses:

1. **Phosphate-residue contacts** (phos MD only)
   For each non-phosphate residue, the fraction of frames in which any
   side-chain heavy atom comes within *cutoff* Å of the phosphate group (P
   atom or phosphate oxygens).  Identifies which residues coordinate the
   phosphate — typically R26 and R33 in the LMNA variants.

2. **Contact maps**
   Per-frame Cα-Cα contact frequency maps (binary contact defined as
   distance < cutoff).  Used for:
     * Time-averaged contact frequency grids (comparing apo vs phos MD)
     * Reference comparison against AF3-predicted contact maps
     * Topology-based clustering (16_contact_map_clustering.py logic)

Algorithms are extracted from 08_analyze_md_local.py, 12_dual_reference_analysis.py,
and 16_contact_map_clustering.py.

Units: distances in Å throughout.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .io_utils import load_trajectory_universe, get_rep_dirs

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Atom selection helper
# ---------------------------------------------------------------------------

def select_protein_ca(universe) -> "mda.AtomGroup":
    """Select Cα atoms including phospho-amino acid variants (PTR, SEP, TPO, TP1, TP2)."""
    return universe.select_atoms(
        "(protein and name CA) or "
        "(resname PTR SEP TPO TP1 TP2 and name CA)"
    )


# ---------------------------------------------------------------------------
# Contact map computation
# ---------------------------------------------------------------------------

def contact_map_from_coords(
    coords: np.ndarray,
    cutoff_A: float = 8.0,
    mask_seq_neighbors: int = 2,
) -> np.ndarray:
    """Compute a binary Cα-Cα contact map from a coordinate array.

    Sequence-adjacent contacts (|i-j| <= *mask_seq_neighbors*) are masked
    to zero because they are trivially present in any protein backbone and
    carry no topological information about folding.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Cα coordinates in Å.
    cutoff_A : float
        Distance threshold for "in contact" in Å. Default 8 Å (standard for
        Cα-Cα contact maps; the upstream scripts use 8 Å).
    mask_seq_neighbors : int
        Mask contacts between residues within this sequence separation.

    Returns
    -------
    cm : np.ndarray, shape (N, N), dtype uint8
        Binary contact map.  cm[i, j] = 1 if residues i and j are in contact.
    """
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]      # (N, N, 3)
    dist = np.sqrt((diff ** 2).sum(axis=2))              # (N, N)
    cm = (dist < cutoff_A).astype(np.uint8)
    np.fill_diagonal(cm, 0)
    for k in range(1, mask_seq_neighbors + 1):
        for i in range(n - k):
            cm[i, i + k] = 0
            cm[i + k, i] = 0
    return cm


def compute_contact_map_traj(
    universe,
    cutoff_A: float = 8.0,
    n_sample: int = 200,
) -> Optional[np.ndarray]:
    """Compute the time-averaged Cα-Cα contact frequency map over a trajectory.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    cutoff_A : float
        Contact cutoff in Å.
    n_sample : int
        Number of evenly-spaced frames to sample.

    Returns
    -------
    contact_freq : np.ndarray, shape (N, N), values in [0, 1]
        Fraction of sampled frames in which each residue pair is in contact.
        Returns None if Cα selection is empty.
    """
    ca = select_protein_ca(universe)
    n = len(ca)
    if n == 0:
        return None

    n_frames = len(universe.trajectory)
    indices = (
        np.linspace(0, n_frames - 1, n_sample, dtype=int)
        if n_sample < n_frames
        else np.arange(n_frames)
    )

    contacts = np.zeros((n, n))
    for fi in indices:
        universe.trajectory[fi]
        pos = ca.positions
        dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
        contacts += (dist < cutoff_A).astype(float)

    return contacts / len(indices)


def compute_contact_map_from_pdb(
    pdb_path: Union[str, Path],
    cutoff_A: float = 8.0,
) -> np.ndarray:
    """Compute binary Cα-Cα contact map from a single PDB file.

    Useful for obtaining AF3 reference contact maps.

    Returns
    -------
    cm : np.ndarray, shape (N, N)
    """
    u = mda.Universe(str(pdb_path))
    ca = select_protein_ca(u)
    return contact_map_from_coords(ca.positions, cutoff_A)


# ---------------------------------------------------------------------------
# Contact-map similarity metrics
# ---------------------------------------------------------------------------

def jaccard_similarity(cm_a: np.ndarray, cm_b: np.ndarray) -> float:
    """Jaccard similarity between two binary contact maps.

    J = |intersection| / |union|.  Returns 1.0 if both maps are all-zeros.

    The Jaccard metric is preferred over Pearson correlation for binary maps
    because it is insensitive to the baseline number of contacts, focusing
    only on the *changed* contacts.
    """
    inter = np.logical_and(cm_a, cm_b).sum()
    uni = np.logical_or(cm_a, cm_b).sum()
    if uni == 0:
        return 1.0
    return float(inter / uni)


def hamming_similarity(cm_a: np.ndarray, cm_b: np.ndarray) -> float:
    """Fraction of equal elements in the upper triangle of two contact maps.

    1 − (normalised Hamming distance).
    """
    n = cm_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a = cm_a[iu]
    b = cm_b[iu]
    return float((a == b).mean())


def cm_to_feature_vec(cm: np.ndarray) -> np.ndarray:
    """Flatten the upper triangle of a contact map to a 1-D float32 feature vector.

    Used for clustering: each frame is represented by a single vector derived
    from its Cα-Cα contact topology.
    """
    n = cm.shape[0]
    iu = np.triu_indices(n, k=1)
    return cm[iu].astype(np.float32)


# ---------------------------------------------------------------------------
# Phosphate-residue contacts
# ---------------------------------------------------------------------------

def compute_phos_contacts_mda(
    universe,
    phos_resnum: int = 30,
    cutoff_A: float = 5.0,
    n_sample: int = 200,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute per-residue phosphate-contact fractions from a trajectory.

    For each non-phosphate residue, measures the fraction of frames in which
    *any* side-chain heavy atom comes within *cutoff_A* Å of the phosphorus
    atom (or phosphate oxygens) on the PTR residue at position *phos_resnum*.

    Apo trajectories (no PTR residue) return ``None``.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    phos_resnum : int
        Residue number of the phospho-tyrosine (PTR).
    cutoff_A : float
        Contact distance threshold in Å.
    n_sample : int
        Frames to sample.

    Returns
    -------
    (residue_ids, contact_fractions) : (np.ndarray, np.ndarray) or None
        residue_ids : 1-D int array of residue IDs (all residues except PTR)
        contact_fractions : 1-D float array, fraction of frames in contact
    """
    # Find phosphate group atoms: P atom (CHARMM naming: P or P1)
    phos_sel = universe.select_atoms(
        f"(resid {phos_resnum} and (name P or name P1 or "
        f"name O1P or name O2P or name O3P or name OP1 or name OP2 or "
        f"name O2 or name O3 or name O4))"
    )
    if len(phos_sel) == 0:
        return None  # Apo trajectory; no phosphate present

    # All protein residues except the phospho site
    protein_residues = [
        r for r in universe.select_atoms("protein").residues
        if r.resid != phos_resnum
    ]
    residue_ids = np.array([r.resid for r in protein_residues])
    n_res = len(protein_residues)

    n_frames = len(universe.trajectory)
    indices = (
        np.linspace(0, n_frames - 1, n_sample, dtype=int)
        if n_sample < n_frames
        else np.arange(n_frames)
    )

    counts = np.zeros(n_res)
    n_done = 0
    for fi in indices:
        universe.trajectory[fi]
        n_done += 1
        p_centroid = phos_sel.positions.mean(axis=0)
        for ri, res in enumerate(protein_residues):
            # Use side-chain heavy atoms; fall back to all atoms if Gly/Pro
            side = res.atoms.select_atoms(
                "not name N and not name CA and not name C and not name O and not name H*"
            )
            if len(side) == 0:
                side = res.atoms
            dists = np.sqrt(((side.positions - p_centroid) ** 2).sum(axis=1))
            if dists.min() < cutoff_A:
                counts[ri] += 1

    fractions = counts / max(1, n_done)
    return residue_ids, fractions


def compute_phos_contacts_from_dir(
    rep_dir: Union[str, Path],
    phos_resnum: int = 30,
    cutoff_A: float = 5.0,
    n_sample: int = 200,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load trajectory from *rep_dir* and compute phosphate-residue contacts."""
    universe = load_trajectory_universe(rep_dir)
    if universe is None:
        return None
    return compute_phos_contacts_mda(universe, phos_resnum, cutoff_A, n_sample)


# ---------------------------------------------------------------------------
# Topology-based clustering (from 16_contact_map_clustering.py)
# ---------------------------------------------------------------------------

def collect_contact_maps(
    candidate: str,
    prod_dir: Union[str, Path],
    n_per_rep: int = 150,
    cutoff_A: float = 8.0,
) -> Tuple[Optional[np.ndarray], Optional[List[dict]]]:
    """Collect subsampled contact maps for all reps of a candidate.

    Pools frames from both apo and phos replicates to build a joint
    conformational landscape.

    Parameters
    ----------
    candidate : str
        Base candidate name (e.g. ``"cand_12345_..."``).
    prod_dir : path
    n_per_rep : int
        Frames to sample per replicate.
    cutoff_A : float
        Cα-Cα contact cutoff.

    Returns
    -------
    cms : np.ndarray, shape (T, N, N), dtype uint8 — or None
    labels : list of dicts {condition, rep, frame, time_ns} — or None
    """
    prod_dir = Path(prod_dir)
    cms_list = []
    labels = []
    n_res = None

    for cond in ("apo", "phos"):
        sys_dir = prod_dir / f"{candidate}_{cond}"
        if not sys_dir.exists():
            continue
        for rep_dir in get_rep_dirs(sys_dir):
            u = load_trajectory_universe(rep_dir)
            if u is None:
                continue
            ca = select_protein_ca(u)
            if n_res is None:
                n_res = len(ca)
            n_frames = len(u.trajectory)
            indices = (
                np.linspace(0, n_frames - 1, n_per_rep, dtype=int)
                if n_per_rep < n_frames
                else np.arange(n_frames)
            )
            for fi in indices:
                u.trajectory[fi]
                coords = ca.positions[:n_res].copy()
                cm = contact_map_from_coords(coords, cutoff_A)
                cms_list.append(cm)
                labels.append({
                    "condition": cond,
                    "rep": rep_dir.name,
                    "frame": int(fi),
                    "time_ns": float(u.trajectory.time) / 1000.0,
                })

    if not cms_list:
        return None, None

    return np.stack(cms_list), labels


def cluster_contact_maps(
    cms: np.ndarray,
    n_clusters: int = 5,
    method: str = "agglomerative",
) -> np.ndarray:
    """Cluster binary contact maps using topology-space features.

    Parameters
    ----------
    cms : np.ndarray, shape (T, N, N)
        Stack of contact maps (integer, 0/1).
    n_clusters : int
        Number of clusters.
    method : {"agglomerative", "kmeans"}
        Clustering algorithm.  Agglomerative (Ward linkage) is the default
        because it does not assume Gaussian clusters and handles binary data
        better than K-means.

    Returns
    -------
    cluster_ids : np.ndarray, shape (T,)
        Cluster label per frame (0-indexed).
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for clustering. "
            "Install with: pip install scikit-learn"
        )

    features = np.array([cm_to_feature_vec(cm) for cm in cms])

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")

    return model.fit_predict(features)


def compute_per_frame_similarities(
    cms: np.ndarray,
    ref_apo_cm: np.ndarray,
    ref_phos_cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-frame Jaccard similarity to each reference contact map.

    Parameters
    ----------
    cms : np.ndarray, shape (T, N, N)
    ref_apo_cm, ref_phos_cm : np.ndarray, shape (N, N)

    Returns
    -------
    sim_apo : np.ndarray, shape (T,)
    sim_phos : np.ndarray, shape (T,)
    """
    sim_apo = np.array([jaccard_similarity(cm, ref_apo_cm) for cm in cms])
    sim_phos = np.array([jaccard_similarity(cm, ref_phos_cm) for cm in cms])
    return sim_apo, sim_phos
