"""Console entry point: ``make-figures``."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    """Generate all paper-quality MD figures from pre-computed aggregated CSVs."""
    ap = argparse.ArgumentParser(
        prog="make-figures",
        description="Generate paper figures from aggregated CSV outputs.",
    )
    ap.add_argument("--analysis-dir", required=True,
                    help="Output dir from analyze-trajectory")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for PDF/PNG figures")
    ap.add_argument("--phos-resnum", type=int, default=30)
    args = ap.parse_args(argv)

    try:
        from phosphoswitch_md.figures import (
            collect_aggregated_pairs,
            plot_rmsd_timeseries_grid,
            plot_rg_timeseries_grid,
            plot_rmsf_grid,
            plot_sasa_grid,
            plot_phos_contact_bar,
        )
        from phosphoswitch_md.io_utils import read_aggregated_csv
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    analysis_dir = Path(args.analysis_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_aggregated_pairs(analysis_dir)
    if not pairs:
        print(f"ERROR: no aggregated data in {analysis_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} candidates.\n")

    def _try(label, fn, *a, **kw):
        try:
            fn(*a, **kw)
            print(f"  OK   {label}")
        except Exception as e:
            print(f"  FAIL {label}: {e}")

    _try("rmsd_vs_time.pdf",
         plot_rmsd_timeseries_grid, pairs, out_dir / "rmsd_vs_time.pdf", args.phos_resnum)
    _try("rg_vs_time.pdf",
         plot_rg_timeseries_grid, pairs, out_dir / "rg_vs_time.pdf")
    _try("rmsf_per_residue.pdf",
         plot_rmsf_grid, pairs, out_dir / "rmsf_per_residue.pdf", args.phos_resnum)
    _try("sasa_y30_vs_time.pdf",
         plot_sasa_grid, pairs, out_dir / "sasa_y30_vs_time.pdf", args.phos_resnum)
    _try("phos_contact_map.pdf",
         plot_phos_contact_bar, pairs, out_dir / "phos_contact_map.pdf", args.phos_resnum)

    # Summary table
    _write_summary_table(pairs, out_dir / "summary_table.csv", read_aggregated_csv, args.phos_resnum)
    print(f"  OK   summary_table.csv")
    print(f"\nAll figures in: {out_dir}")


def _write_summary_table(pairs, out_csv, read_aggregated_csv, phos_resnum):
    rows = []
    for tag in sorted(pairs.keys()):
        for cond in ("apo", "phos"):
            agg = pairs[tag].get(cond)
            if not agg:
                continue
            row = {"candidate": tag, "condition": cond}
            for metric, fname, col, factor, out_col in [
                ("rmsd", "rmsd_mean_sem.csv", "rmsd_mean_nm", 10, "rmsd_final_A"),
                ("rg",   "rg_mean_sem.csv",   "rg_mean_nm",   10, "rg_final_A"),
                ("sasa", "sasa_mean_sem.csv",  "sasa_mean_nm2", 100, f"sasa_y{phos_resnum}_final_A2"),
            ]:
                data = read_aggregated_csv(agg / fname)
                if data is not None:
                    n = len(data[col])
                    tail = max(1, n // 5)
                    row[out_col] = f"{float(np.mean(data[col][-tail:])) * factor:.2f}"
                else:
                    row[out_col] = ""
            rmsf_data = read_aggregated_csv(agg / "rmsf_mean_sem.csv")
            row["rmsf_mean_A"] = (
                f"{float(np.mean(rmsf_data['rmsf_mean_nm'])) * 10:.2f}"
                if rmsf_data is not None else ""
            )
            rows.append(row)
    if rows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
