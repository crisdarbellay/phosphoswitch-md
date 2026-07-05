"""
figures.py — Paper-quality plots for the phosphoswitch MD analysis.

Implements all figure types from the upstream pipeline scripts
(09_paper_figures_local.py, 12_dual_reference_analysis.py,
14_basin_occupancy_analysis.py, 16_contact_map_clustering.py).

Color convention (consistent with upstream scripts):
  Apo MD trajectories  → #2E86AB  (blue)
  Phos MD trajectories → #A23B72  (magenta/pink)
  Apo AF3 reference    → #0d3b66  (dark blue)
  Phos AF3 reference   → #7b1fa2  (dark purple)

All output functions write a file and return its Path.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .io_utils import read_aggregated_csv

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
APO_COLOR = "#2E86AB"
PHOS_COLOR = "#A23B72"
COL_APO_TRAJ = "#2c7fb8"
COL_PHOS_TRAJ = "#c994c7"
COL_APO_REF = "#0d3b66"
COL_PHOS_REF = "#7b1fa2"

_COND_COLORS = {"apo": APO_COLOR, "phos": PHOS_COLOR}
_REP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ---------------------------------------------------------------------------
# Helpers for aggregated-CSV figures (from 09_paper_figures_local.py)
# ---------------------------------------------------------------------------

def _parse_system_name(name: str) -> Tuple[str, str]:
    """Split e.g. 'cand_XXX_apo' → ('cand_XXX', 'apo')."""
    if name.endswith("_phos"):
        return name[:-5], "phos"
    if name.endswith("_apo"):
        return name[:-4], "apo"
    return name, "unknown"


def collect_aggregated_pairs(analysis_dir: Union[str, Path]) -> Dict[str, Dict[str, Path]]:
    """Discover all system aggregated directories in *analysis_dir*.

    Returns a dict keyed by candidate tag, values are ``{cond: agg_path}``.
    """
    analysis_dir = Path(analysis_dir)
    pairs: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for sys_dir in analysis_dir.iterdir():
        if not sys_dir.is_dir():
            continue
        agg = sys_dir / "aggregated"
        if not agg.is_dir():
            continue
        tag, cond = _parse_system_name(sys_dir.name)
        pairs[tag][cond] = agg
    return dict(pairs)


def _setup_grid(n: int, ncols: int = 3) -> Tuple[plt.Figure, np.ndarray]:
    """Create a grid of axes with ``n`` panels and a maximum of ``ncols`` columns."""
    nrows = (n + ncols - 1) // ncols
    ncols_actual = min(ncols, n)
    fig, axes = plt.subplots(
        nrows, ncols_actual, figsize=(5 * ncols_actual, 3 * nrows),
        squeeze=False, sharex=False,
    )
    return fig, axes


def _hide_empty_axes(axes: np.ndarray, n: int) -> None:
    nrows, ncols = axes.shape
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)


# ---------------------------------------------------------------------------
# 09-style paper grid figures (read pre-aggregated CSVs)
# ---------------------------------------------------------------------------

def plot_rmsd_timeseries_grid(
    pairs: Dict[str, Dict[str, Path]],
    out_path: Union[str, Path],
    phos_resnum: int = 30,
) -> Path:
    """Cα RMSD vs time, apo (blue) vs phos (magenta), one panel per candidate.

    Reads ``rmsd_mean_sem.csv`` from each ``aggregated/`` directory.
    Values are converted nm → Å for display.

    Parameters
    ----------
    pairs : dict
        Output of :func:`collect_aggregated_pairs`.
    out_path : path
        Output PDF/PNG file.
    phos_resnum : int
        Phospho-site residue number (for annotation only).

    Returns
    -------
    Path to the written file.
    """
    candidates = sorted(pairs.keys())
    n = len(candidates)
    if n == 0:
        raise ValueError("No candidates found.")

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                              squeeze=False, sharex=False)
    fig.suptitle("Cα RMSD vs time", fontsize=13, fontweight="bold")

    for i, tag in enumerate(candidates):
        ax = axes[i // ncols][i % ncols]
        for cond, color in [("apo", APO_COLOR), ("phos", PHOS_COLOR)]:
            agg = pairs[tag].get(cond)
            if not agg:
                continue
            data = read_aggregated_csv(agg / "rmsd_mean_sem.csv")
            if data is None:
                continue
            t = data["time_ns"]
            m = data["rmsd_mean_nm"] * 10   # nm → Å
            s = data["rmsd_sem_nm"] * 10
            ax.plot(t, m, color=color, lw=1.5, label=cond)
            ax.fill_between(t, m - s, m + s, color=color, alpha=0.2)

        ax.set_title(tag[:35], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Time (ns)")
        if i % ncols == 0:
            ax.set_ylabel("Cα RMSD (Å)")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    _hide_empty_axes(axes, n)
    plt.tight_layout()
    out_path = Path(out_path)
    ext = out_path.suffix.lstrip(".") or "pdf"
    plt.savefig(out_path, format=ext, bbox_inches="tight")
    plt.close()
    return out_path


def plot_rg_timeseries_grid(
    pairs: Dict[str, Dict[str, Path]],
    out_path: Union[str, Path],
) -> Path:
    """Radius of gyration vs time, apo vs phos.  Values converted nm → Å."""
    candidates = sorted(pairs.keys())
    n = len(candidates)
    if n == 0:
        raise ValueError("No candidates found.")

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                              squeeze=False, sharex=False)
    fig.suptitle("Radius of gyration vs time", fontsize=13, fontweight="bold")

    for i, tag in enumerate(candidates):
        ax = axes[i // ncols][i % ncols]
        for cond, color in [("apo", APO_COLOR), ("phos", PHOS_COLOR)]:
            agg = pairs[tag].get(cond)
            if not agg:
                continue
            data = read_aggregated_csv(agg / "rg_mean_sem.csv")
            if data is None:
                continue
            t = data["time_ns"]
            m = data["rg_mean_nm"] * 10
            s = data["rg_sem_nm"] * 10
            ax.plot(t, m, color=color, lw=1.5, label=cond)
            ax.fill_between(t, m - s, m + s, color=color, alpha=0.2)

        ax.set_title(tag[:35], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Time (ns)")
        if i % ncols == 0:
            ax.set_ylabel("Rg (Å)")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    _hide_empty_axes(axes, n)
    plt.tight_layout()
    out_path = Path(out_path)
    ext = out_path.suffix.lstrip(".") or "pdf"
    plt.savefig(out_path, format=ext, bbox_inches="tight")
    plt.close()
    return out_path


def plot_rmsf_grid(
    pairs: Dict[str, Dict[str, Path]],
    out_path: Union[str, Path],
    phos_resnum: int = 30,
) -> Path:
    """Per-residue Cα RMSF, apo vs phos, with phospho-site annotation."""
    candidates = sorted(pairs.keys())
    n = len(candidates)
    if n == 0:
        raise ValueError("No candidates found.")

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                              squeeze=False, sharex=False)
    fig.suptitle("Per-residue Cα RMSF (last 80% of trajectory)",
                 fontsize=13, fontweight="bold")

    for i, tag in enumerate(candidates):
        ax = axes[i // ncols][i % ncols]
        for cond, color in [("apo", APO_COLOR), ("phos", PHOS_COLOR)]:
            agg = pairs[tag].get(cond)
            if not agg:
                continue
            data = read_aggregated_csv(agg / "rmsf_mean_sem.csv")
            if data is None:
                continue
            r = data["residue"]
            m = data["rmsf_mean_nm"] * 10
            s = data["rmsf_sem_nm"] * 10
            ax.plot(r, m, color=color, lw=1.5, label=cond)
            ax.fill_between(r, m - s, m + s, color=color, alpha=0.2)

        ax.axvline(phos_resnum, color="orange", lw=0.8, ls="--", alpha=0.7,
                   label=f"Y{phos_resnum}")
        ax.set_title(tag[:35], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Residue")
        if i % ncols == 0:
            ax.set_ylabel("RMSF (Å)")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    _hide_empty_axes(axes, n)
    plt.tight_layout()
    out_path = Path(out_path)
    ext = out_path.suffix.lstrip(".") or "pdf"
    plt.savefig(out_path, format=ext, bbox_inches="tight")
    plt.close()
    return out_path


def plot_sasa_grid(
    pairs: Dict[str, Dict[str, Path]],
    out_path: Union[str, Path],
    phos_resnum: int = 30,
) -> Path:
    """SASA of the phospho-tyrosine sidechain over time.  Values converted nm² → Å²."""
    candidates = sorted(pairs.keys())
    n = len(candidates)
    if n == 0:
        raise ValueError("No candidates found.")

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                              squeeze=False, sharex=False)
    fig.suptitle(f"Sidechain SASA of Y{phos_resnum}/PTR over time",
                 fontsize=13, fontweight="bold")

    for i, tag in enumerate(candidates):
        ax = axes[i // ncols][i % ncols]
        for cond, color in [("apo", APO_COLOR), ("phos", PHOS_COLOR)]:
            agg = pairs[tag].get(cond)
            if not agg:
                continue
            data = read_aggregated_csv(agg / "sasa_mean_sem.csv")
            if data is None:
                continue
            t = data["time_ns"]
            m = data["sasa_mean_nm2"] * 100   # nm² → Å²
            s = data["sasa_sem_nm2"] * 100
            ax.plot(t, m, color=color, lw=1.2, label=cond)
            ax.fill_between(t, m - s, m + s, color=color, alpha=0.2)

        ax.set_title(tag[:35], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Time (ns)")
        if i % ncols == 0:
            ax.set_ylabel(f"SASA Y{phos_resnum} (Å²)")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    _hide_empty_axes(axes, n)
    plt.tight_layout()
    out_path = Path(out_path)
    ext = out_path.suffix.lstrip(".") or "pdf"
    plt.savefig(out_path, format=ext, bbox_inches="tight")
    plt.close()
    return out_path


def plot_phos_contact_bar(
    pairs: Dict[str, Dict[str, Path]],
    out_path: Union[str, Path],
    phos_resnum: int = 30,
) -> Path:
    """Per-residue phosphate-contact persistence bar chart (phos simulations only)."""
    rows = []
    for tag in sorted(pairs.keys()):
        agg = pairs[tag].get("phos")
        if not agg:
            continue
        data = read_aggregated_csv(agg / "phos_contacts_mean_sem.csv")
        if data is None:
            continue
        rows.append((tag, data["residue"], data["contact_fraction_mean"]))

    if not rows:
        raise ValueError("No phosphate-contact data found.")

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (tag, residues, fractions) in zip(axes, rows):
        ax.bar(residues, fractions, color=PHOS_COLOR, width=0.8, alpha=0.85)
        ax.axvline(phos_resnum, color="orange", lw=0.8, ls="--", alpha=0.7,
                   label=f"Y{phos_resnum}/PTR")
        ax.set_title(tag[:50], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Frac.")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Residue")
    plt.tight_layout()
    out_path = Path(out_path)
    ext = out_path.suffix.lstrip(".") or "pdf"
    plt.savefig(out_path, format=ext, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Dual-RMSD scatter plot (from 12_dual_reference_analysis.py)
# ---------------------------------------------------------------------------

def plot_dual_rmsd_scatter(
    cand_data: Dict[str, Dict[str, dict]],
    candidate: str,
    out_dir: Union[str, Path],
) -> Path:
    """Three-panel scatter plot of RMSD to apo vs RMSD to phos reference.

    Panels: [apo MD] [phos MD] [overlay].
    Star markers indicate the mean position of the last 10% of each replicate.

    Parameters
    ----------
    cand_data : dict
        ``{cond: {rep_name: {rmsd_to_apo, rmsd_to_phos, time_ns}}}``
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{candidate}: trajectory distance to reference states",
                 fontsize=13, fontweight="bold")

    for ax_i, (cond, title, traj_color, ref_color) in enumerate([
        ("apo", "Apo MD", COL_APO_TRAJ, COL_APO_REF),
        ("phos", "Phos MD", COL_PHOS_TRAJ, COL_PHOS_REF),
    ]):
        ax = axes[ax_i]
        if cond in cand_data:
            for rep, data in cand_data[cond].items():
                ax.scatter(data["rmsd_to_apo"], data["rmsd_to_phos"],
                           s=2, alpha=0.2, c=traj_color,
                           label=rep if rep == list(cand_data[cond].keys())[0] else None)
            # Mark mean of last 10%
            for rep, data in cand_data[cond].items():
                last_n = max(1, int(0.1 * len(data["time_ns"])))
                ax.scatter(
                    data["rmsd_to_apo"][-last_n:].mean(),
                    data["rmsd_to_phos"][-last_n:].mean(),
                    s=80, c=ref_color, marker="*", edgecolor="black",
                    linewidth=1, zorder=10
                )

        ax.set_xlabel("Cα RMSD to apo reference (Å)")
        ax.set_ylabel("Cα RMSD to phos reference (Å)")
        ax.set_title(title)
        ax.plot([0, 30], [0, 30], "k--", alpha=0.3, lw=1, label="equidistant")
        ax.set_xlim(0, max(20, ax.get_xlim()[1]))
        ax.set_ylim(0, max(20, ax.get_ylim()[1]))
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    # Overlay panel
    ax = axes[2]
    for cond, traj_color in [("apo", COL_APO_TRAJ), ("phos", COL_PHOS_TRAJ)]:
        if cond in cand_data:
            x, y = [], []
            for rep, data in cand_data[cond].items():
                x.extend(data["rmsd_to_apo"])
                y.extend(data["rmsd_to_phos"])
            ax.scatter(x, y, s=2, alpha=0.15, c=traj_color, label=cond)

    ax.set_xlabel("Cα RMSD to apo ref (Å)")
    ax.set_ylabel("Cα RMSD to phos ref (Å)")
    ax.set_title("Overlay (apo blue, phos pink)")
    ax.plot([0, 30], [0, 30], "k--", alpha=0.3, lw=1)
    ax.set_xlim(0, max(20, ax.get_xlim()[1]))
    ax.set_ylim(0, max(20, ax.get_ylim()[1]))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_dual_rmsd_scatter.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Basin occupancy bar chart (from 14_basin_occupancy_analysis.py)
# ---------------------------------------------------------------------------

def plot_occupancy_bars(
    apo_metrics: Dict[str, float],
    phos_metrics: Dict[str, float],
    candidate: str,
    out_dir: Union[str, Path],
    strict_A: float = 4.0,
    loose_A: float = 7.0,
) -> Path:
    """Grouped bar chart comparing basin occupancy fractions for apo vs phos MD.

    Shows strict (4 Å) and loose (7 Å) threshold panels side by side.

    Parameters
    ----------
    apo_metrics, phos_metrics : dict
        Output of :func:`~phosphoswitch_md.basin.compute_occupancy_metrics`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{candidate}: conformational basin occupancy", fontweight="bold")

    for ax, (label, key) in zip(axes, [
        (f"strict (<{strict_A} Å)", "strict"),
        (f"loose (<{loose_A} Å)", "loose"),
    ]):
        if key == "strict":
            cats = ["apo basin", "phos basin", "neither"]
            apo_vals = [apo_metrics["frac_apo_strict"],
                        apo_metrics["frac_phos_strict"],
                        max(0.0, 1 - apo_metrics["frac_apo_strict"] - apo_metrics["frac_phos_strict"])]
            phos_vals = [phos_metrics["frac_apo_strict"],
                         phos_metrics["frac_phos_strict"],
                         max(0.0, 1 - phos_metrics["frac_apo_strict"] - phos_metrics["frac_phos_strict"])]
        else:
            cats = ["apo basin\n(excl. both)", "phos basin\n(excl. both)", "neither", "both basins"]
            apo_vals = [
                max(0.0, apo_metrics["frac_apo_loose"] - apo_metrics["frac_both_loose"]),
                max(0.0, apo_metrics["frac_phos_loose"] - apo_metrics["frac_both_loose"]),
                apo_metrics["frac_neither_loose"],
                apo_metrics["frac_both_loose"],
            ]
            phos_vals = [
                max(0.0, phos_metrics["frac_apo_loose"] - phos_metrics["frac_both_loose"]),
                max(0.0, phos_metrics["frac_phos_loose"] - phos_metrics["frac_both_loose"]),
                phos_metrics["frac_neither_loose"],
                phos_metrics["frac_both_loose"],
            ]

        x = np.arange(len(cats))
        w = 0.35
        ax.bar(x - w / 2, apo_vals, w, label="apo MD",
               color=APO_COLOR, edgecolor="black", linewidth=0.5)
        ax.bar(x + w / 2, phos_vals, w, label="phos MD",
               color=PHOS_COLOR, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylabel("Fraction of frames")
        ax.set_title(f"Threshold: {label}")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, max(1.05, max(apo_vals + phos_vals) * 1.15))

        # Annotate phos-basin enrichment (index 1 = phos basin)
        eps = 1e-4
        enrich = (phos_vals[1] + eps) / (apo_vals[1] + eps)
        top_y = max(apo_vals[1], phos_vals[1]) + 0.07
        ax.text(1, top_y, f"×{enrich:.1f} enrichment",
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_occupancy_bars.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# 2D free-energy landscape (from 14_basin_occupancy_analysis.py)
# ---------------------------------------------------------------------------

def plot_landscape_2d(
    cand_data: Dict[str, List[dict]],
    candidate: str,
    out_dir: Union[str, Path],
    strict_A: float = 4.0,
    loose_A: float = 7.0,
    x_max: float = 30.0,
    y_max: float = 30.0,
) -> Path:
    """2D −log(P) free-energy landscape on (RMSD_to_apo, RMSD_to_phos).

    Parameters
    ----------
    cand_data : dict
        ``{cond: [{"rmsd_apo": ..., "rmsd_phos": ...}, ...]}`` (one dict per rep).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        f"{candidate}: 2D conformational landscape\n"
        f"strict = <{strict_A}Å, loose = <{loose_A}Å",
        fontsize=12, fontweight="bold"
    )

    n_bins = 40
    edges_x = np.linspace(0, x_max, n_bins + 1)
    edges_y = np.linspace(0, y_max, n_bins + 1)

    for ax, cond, title in [(axes[0], "apo", "Apo MD"), (axes[1], "phos", "Phos MD")]:
        if cond not in cand_data or not cand_data[cond]:
            ax.text(0.5, 0.5, f"No {cond} data",
                    transform=ax.transAxes, ha="center")
            continue

        x_all = np.concatenate([d["rmsd_apo"] for d in cand_data[cond]])
        y_all = np.concatenate([d["rmsd_phos"] for d in cand_data[cond]])

        H, _, _ = np.histogram2d(x_all, y_all, bins=[edges_x, edges_y])
        P = H / H.sum()
        eps = 1.0 / (n_bins ** 2) * 0.01
        F = -np.log(P + eps)
        F -= F.min()

        im = ax.imshow(
            F.T, origin="lower",
            extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]],
            cmap="viridis_r", aspect="auto", vmax=8
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label("−log P (relative free energy)")

        # Basin threshold lines
        for thr, color in [(strict_A, "white"), (loose_A, "yellow")]:
            ax.axvline(thr, color=color, lw=0.8, ls="--", alpha=0.7)
            ax.axhline(thr, color=color, lw=0.8, ls="--", alpha=0.7)

        ax.plot([0, max(x_max, y_max)], [0, max(x_max, y_max)],
                "w--", lw=0.5, alpha=0.5)
        ax.set_xlabel("RMSD to AF3 apo ref (Å)")
        ax.set_ylabel("RMSD to AF3 phos ref (Å)")
        ax.set_title(title)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_landscape_2d.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# RMSD time-series with min-visit markers (from 14_)
# ---------------------------------------------------------------------------

def plot_min_rmsd_timeseries(
    cand_data: Dict[str, List[dict]],
    candidate: str,
    out_dir: Union[str, Path],
) -> Path:
    """Time-resolved RMSD to each reference per replicate, with minimum markers.

    The minimum-RMSD point is highlighted with a scatter marker to show
    the closest approach to each reference basin.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=False)
    fig.suptitle(f"{candidate}: time-resolved RMSD to each reference",
                 fontweight="bold")

    for ci, cond in enumerate(["apo", "phos"]):
        if cond not in cand_data:
            continue
        for ri, rep_data in enumerate(cand_data[cond]):
            t = rep_data["time_ns"]
            r_apo = rep_data["rmsd_apo"]
            r_phos = rep_data["rmsd_phos"]
            color = _REP_COLORS[ri % len(_REP_COLORS)]
            rep_label = rep_data.get("rep", f"rep{ri + 1}")

            ax = axes[0, ci]
            ax.plot(t, r_apo, lw=0.6, alpha=0.7, color=color, label=rep_label)
            min_i = int(np.argmin(r_apo))
            ax.scatter(t[min_i], r_apo[min_i], s=40, c=color,
                       edgecolor="black", linewidth=0.8, zorder=10)

            ax = axes[1, ci]
            ax.plot(t, r_phos, lw=0.6, alpha=0.7, color=color, label=rep_label)
            min_i = int(np.argmin(r_phos))
            ax.scatter(t[min_i], r_phos[min_i], s=40, c=color,
                       edgecolor="black", linewidth=0.8, zorder=10)

        axes[0, ci].set_title(f"{cond.upper()} MD")
        axes[0, ci].set_ylabel("RMSD to apo ref (Å)")
        axes[0, ci].legend(loc="upper right", fontsize=8)
        axes[0, ci].grid(alpha=0.3)
        axes[0, ci].axhline(4, color="green", linestyle=":", lw=0.8, label="4Å")
        axes[0, ci].axhline(7, color="orange", linestyle=":", lw=0.8, label="7Å")

        axes[1, ci].set_ylabel("RMSD to phos ref (Å)")
        axes[1, ci].set_xlabel("Time (ns)")
        axes[1, ci].legend(loc="upper right", fontsize=8)
        axes[1, ci].grid(alpha=0.3)
        axes[1, ci].axhline(4, color="green", linestyle=":", lw=0.8)
        axes[1, ci].axhline(7, color="orange", linestyle=":", lw=0.8)

    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_min_rmsd_timeseries.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Contact-map grid (from 12_dual_reference_analysis.py)
# ---------------------------------------------------------------------------

def plot_contact_maps_grid(
    cand_data: Dict[str, List[np.ndarray]],
    candidate: str,
    out_dir: Union[str, Path],
    ref_apo_cm: Optional[np.ndarray] = None,
    ref_phos_cm: Optional[np.ndarray] = None,
    cutoff_A: float = 8.0,
) -> Path:
    """6-panel contact map grid.

    Row 1: MD trajectory averages (apo / phos / Δphos−apo)
    Row 2: AF3 reference contact maps (apo / phos / Δphos−apo)

    Parameters
    ----------
    cand_data : dict
        ``{"apo": [cm_rep1, ...], "phos": [cm_rep1, ...]}``
        Each element is a (N, N) contact frequency array.
    """
    # Determine common size
    sizes = []
    for cond in ("apo", "phos"):
        for cm in cand_data.get(cond, []):
            if cm is not None:
                sizes.append(cm.shape[0])
    if ref_apo_cm is not None:
        sizes.append(ref_apo_cm.shape[0])
    if ref_phos_cm is not None:
        sizes.append(ref_phos_cm.shape[0])

    if not sizes:
        raise ValueError("No contact map data available.")

    n_common = min(sizes)

    def _t(cm):
        return cm[:n_common, :n_common] if cm is not None else None

    apo_maps = [_t(cm) for cm in cand_data.get("apo", []) if cm is not None]
    phos_maps = [_t(cm) for cm in cand_data.get("phos", []) if cm is not None]
    ref_a = _t(ref_apo_cm)
    ref_p = _t(ref_phos_cm)

    avg_apo = np.mean(apo_maps, axis=0) if apo_maps else None
    avg_phos = np.mean(phos_maps, axis=0) if phos_maps else None

    fig = plt.figure(figsize=(15, 10))
    cmap_freq = "viridis"
    cmap_diff = "RdBu_r"

    for idx, (data, title) in enumerate([
        (avg_apo, f"Apo MD ({len(apo_maps)} reps)"),
        (avg_phos, f"Phos MD ({len(phos_maps)} reps)"),
        (None, "Δ phos − apo (MD)"),
        (ref_a, "AF3 apo (predicted)"),
        (ref_p, "AF3 phos (predicted)"),
        (None, "Δ phos − apo (AF3)"),
    ]):
        ax = fig.add_subplot(2, 3, idx + 1)
        if idx == 2:  # MD diff
            if avg_apo is not None and avg_phos is not None:
                delta = avg_phos - avg_apo
                norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
                im = ax.imshow(delta, cmap=cmap_diff, norm=norm, origin="lower")
                plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Δ phos − apo MD\n(red=gained, blue=lost)")
        elif idx == 5:  # AF3 diff
            if ref_a is not None and ref_p is not None:
                delta = ref_p - ref_a
                norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
                im = ax.imshow(delta, cmap=cmap_diff, norm=norm, origin="lower")
                plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Δ phos − apo AF3")
        elif data is not None:
            im = ax.imshow(data, cmap=cmap_freq, vmin=0, vmax=1, origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(title)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")

    plt.suptitle(
        f"{candidate}: Cα-Cα contact maps (cutoff {cutoff_A} Å, N={n_common} residues)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_contact_maps_grid.png"
    plt.savefig(out_file, dpi=110, bbox_inches="tight")
    plt.close()
    return out_file


# ---------------------------------------------------------------------------
# Topology similarity timeseries (from 16_contact_map_clustering.py)
# ---------------------------------------------------------------------------

def plot_topology_similarity(
    labels: List[dict],
    sim_apo: np.ndarray,
    sim_phos: np.ndarray,
    candidate: str,
    out_dir: Union[str, Path],
) -> Path:
    """Per-rep Jaccard similarity to AF3 apo and phos contact maps over time."""
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[(lab["condition"], lab["rep"])].append(i)

    n_groups = len(groups)
    ncols = 3
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows),
                              sharex=False, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for ax_i, ((cond, rep), idxs) in enumerate(sorted(groups.items())):
        ax = axes[ax_i // ncols, ax_i % ncols]
        idxs_sorted = sorted(idxs, key=lambda i: labels[i]["time_ns"])
        times = [labels[i]["time_ns"] for i in idxs_sorted]
        sa = [sim_apo[i] for i in idxs_sorted]
        sp = [sim_phos[i] for i in idxs_sorted]
        ax.plot(times, sa, color=COL_APO_TRAJ, lw=1.0, label="sim → apo ref")
        ax.plot(times, sp, color=COL_PHOS_TRAJ, lw=1.0, label="sim → phos ref")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Jaccard similarity")
        title_color = COL_APO_TRAJ if cond == "apo" else COL_PHOS_TRAJ
        ax.set_title(f"{cond} / {rep}", color=title_color, fontweight="bold")
        ax.grid(alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=8)

    for ax_i in range(n_groups, nrows * ncols):
        axes[ax_i // ncols, ax_i % ncols].set_visible(False)

    plt.suptitle(f"{candidate}: contact-map similarity to AF3 references",
                 fontweight="bold", y=1.00)
    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_topology_similarity.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    return out_file


def plot_cluster_occupancy_bars(
    labels: List[dict],
    cluster_ids: np.ndarray,
    candidate: str,
    out_dir: Union[str, Path],
    n_clusters: int = 5,
) -> Path:
    """Per-condition cluster occupancy bar chart with log2 enrichment annotations."""
    n_apo = sum(1 for l in labels if l["condition"] == "apo")
    n_phos = sum(1 for l in labels if l["condition"] == "phos")
    apo_counts = np.zeros(n_clusters)
    phos_counts = np.zeros(n_clusters)
    for i, lab in enumerate(labels):
        c = cluster_ids[i]
        if lab["condition"] == "apo":
            apo_counts[c] += 1
        else:
            phos_counts[c] += 1

    apo_frac = apo_counts / max(n_apo, 1)
    phos_frac = phos_counts / max(n_phos, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_clusters)
    w = 0.38
    ax.bar(x - w / 2, apo_frac, w, label="apo MD",
           color=APO_COLOR, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, phos_frac, w, label="phos MD",
           color=PHOS_COLOR, edgecolor="black", linewidth=0.5)

    eps = 1e-3
    for i in range(n_clusters):
        log2 = float(np.log2((phos_frac[i] + eps) / (apo_frac[i] + eps)))
        color = "red" if log2 > 0.5 else ("blue" if log2 < -0.5 else "gray")
        ax.text(i, max(apo_frac[i], phos_frac[i]) + 0.02,
                f"log₂={log2:+.1f}", ha="center", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(n_clusters)])
    ax.set_ylabel("Fraction of frames")
    ax.set_title(
        f"{candidate}: topology cluster occupancy by condition\n"
        "log₂(phos/apo) > 0.5 = phos-enriched (red); < −0.5 = apo-enriched (blue)"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_cluster_occupancy.png"
    plt.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close()
    return out_file


def plot_cluster_avg_maps(
    cms: np.ndarray,
    cluster_ids: np.ndarray,
    candidate: str,
    out_dir: Union[str, Path],
    n_clusters: int = 5,
    ref_apo_cm: Optional[np.ndarray] = None,
    ref_phos_cm: Optional[np.ndarray] = None,
) -> Path:
    """Average contact map per cluster alongside AF3 reference maps."""
    fig = plt.figure(figsize=(3.5 * max(n_clusters, 2), 6.5))

    if ref_apo_cm is not None and ref_phos_cm is not None:
        ax = plt.subplot(2, max(n_clusters, 2), 1)
        ax.imshow(ref_apo_cm, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title("AF3 apo", fontweight="bold", color=COL_APO_REF)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")

        ax = plt.subplot(2, max(n_clusters, 2), 2)
        ax.imshow(ref_phos_cm, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title("AF3 phos", fontweight="bold", color=COL_PHOS_REF)
        ax.set_xlabel("Residue")

    for c in range(n_clusters):
        mask = cluster_ids == c
        if not mask.any():
            continue
        avg = cms[mask].mean(axis=0)
        ax = plt.subplot(2, max(n_clusters, 2), n_clusters + c + 1)
        ax.imshow(avg, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"C{c} (n={int(mask.sum())})", fontweight="bold")
        ax.set_xlabel("Residue")
        if c == 0:
            ax.set_ylabel("Residue")

    plt.suptitle(f"{candidate}: average contact maps per cluster vs AF3 refs",
                 fontweight="bold", y=1.0)
    plt.tight_layout()
    out_file = Path(out_dir) / f"{candidate}_cluster_avg_maps.png"
    plt.savefig(out_file, dpi=110, bbox_inches="tight")
    plt.close()
    return out_file
