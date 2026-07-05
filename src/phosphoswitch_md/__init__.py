"""
phosphoswitch-md — MD simulation analysis for phosphoswitch publication.

Validates computationally that LMNA Y45-region designed variants
visit both the straight-helix (apo) and bent-hairpin (phos) conformational
states in unbiased 600 ns GROMACS simulations.

Modules
-------
trajectory
    Cα RMSD vs time, RMSF per residue, radius of gyration, DSSP secondary
    structure.  Both pure-MDAnalysis and GROMACS-wrapper implementations.

basin
    Dual-reference state assignment: per-frame Kabsch-aligned RMSD to both
    AF3 reference structures, basin occupancy fractions, enrichment ratios,
    and first-passage times to the phos basin.

contacts
    Phosphate-residue contact fractions (phos MD only) and Cα-Cα contact
    map analysis.  Includes topology-based clustering (Jaccard + agglomerative).

figures
    Paper-quality plots: RMSD timeseries grids, RMSF grids, basin occupancy
    bar charts, 2D free-energy landscapes, contact map grids, cluster maps.

io_utils
    GROMACS .xvg parsing, trajectory loading, candidate discovery, aggregation
    of multi-replicate time-series to mean ± SEM.

Typical workflow
----------------
1. Postprocess raw GROMACS outputs (strip water, fit Cα) with the
   ``postprocess`` CLI script.
2. Run per-replicate analyses (RMSD, Rg, RMSF, SASA, phosphate contacts)
   with the ``analyze-trajectory`` CLI script.
3. Compute dual-reference basin occupancy with the ``basin-occupancy`` CLI.
4. Generate all paper figures with the ``make-figures`` CLI.
"""

__version__ = "0.1.0"
__author__ = "Cris Darbellay"

from .trajectory import (
    select_protein_ca,
    compute_rmsd_mda,
    compute_rmsf_mda,
    compute_rg_mda,
    compute_dssp_phi_psi,
    run_gmx_rms,
    run_gmx_rg,
    run_gmx_rmsf,
    run_gmx_dssp,
    run_gmx_sasa,
)

from .basin import (
    kabsch_rmsd,
    load_reference_coords,
    compute_dual_rmsd_series,
    compute_dual_rmsd_from_dir,
    assign_basin_per_frame,
    compute_occupancy_metrics,
    find_hitting_time,
    compute_switch_metrics,
    analyze_candidate_basin_occupancy,
)

from .contacts import (
    contact_map_from_coords,
    compute_contact_map_traj,
    compute_contact_map_from_pdb,
    jaccard_similarity,
    hamming_similarity,
    cm_to_feature_vec,
    compute_phos_contacts_mda,
    compute_phos_contacts_from_dir,
    collect_contact_maps,
    cluster_contact_maps,
    compute_per_frame_similarities,
)

from .io_utils import (
    parse_xvg,
    parse_xvg_multi,
    parse_dssp_xpm,
    read_aggregated_csv,
    aggregate_replicates,
    write_mean_sem_csv,
    find_candidates,
    get_rep_dirs,
    load_trajectory_universe,
    find_af3_references,
    run_gmx,
)

__all__ = [
    # trajectory
    "select_protein_ca",
    "compute_rmsd_mda",
    "compute_rmsf_mda",
    "compute_rg_mda",
    "compute_dssp_phi_psi",
    "run_gmx_rms",
    "run_gmx_rg",
    "run_gmx_rmsf",
    "run_gmx_dssp",
    "run_gmx_sasa",
    # basin
    "kabsch_rmsd",
    "load_reference_coords",
    "compute_dual_rmsd_series",
    "compute_dual_rmsd_from_dir",
    "assign_basin_per_frame",
    "compute_occupancy_metrics",
    "find_hitting_time",
    "compute_switch_metrics",
    "analyze_candidate_basin_occupancy",
    # contacts
    "contact_map_from_coords",
    "compute_contact_map_traj",
    "compute_contact_map_from_pdb",
    "jaccard_similarity",
    "hamming_similarity",
    "cm_to_feature_vec",
    "compute_phos_contacts_mda",
    "compute_phos_contacts_from_dir",
    "collect_contact_maps",
    "cluster_contact_maps",
    "compute_per_frame_similarities",
    # io_utils
    "parse_xvg",
    "parse_xvg_multi",
    "parse_dssp_xpm",
    "read_aggregated_csv",
    "aggregate_replicates",
    "write_mean_sem_csv",
    "find_candidates",
    "get_rep_dirs",
    "load_trajectory_universe",
    "find_af3_references",
    "run_gmx",
]
