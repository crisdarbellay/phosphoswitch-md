"""Console entry point: ``analyze-trajectory``."""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    """Run per-replicate RMSD, Rg, RMSF, SASA, and phosphate-contact analyses."""
    ap = argparse.ArgumentParser(
        prog="analyze-trajectory",
        description="Compute RMSD, Rg, RMSF, SASA, and phosphate contacts for all systems.",
    )
    ap.add_argument("--prod-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gmx", default="gmx")
    ap.add_argument("--phos-resnum", type=int, default=30)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--rmsf-start-ps", type=int, default=120_000)
    args = ap.parse_args(argv)

    # Import here to give a clean error if package is not installed
    try:
        from phosphoswitch_md.io_utils import (
            parse_xvg, aggregate_replicates, write_mean_sem_csv,
        )
        from phosphoswitch_md.trajectory import (
            run_gmx_rms, run_gmx_rg, run_gmx_rmsf, run_gmx_sasa,
        )
        from phosphoswitch_md.contacts import compute_phos_contacts_from_dir
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    prod_dir = Path(args.prod_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gmx = shutil.which(args.gmx) or args.gmx

    systems = sorted(d for d in prod_dir.iterdir() if d.is_dir())
    if not systems:
        print(f"ERROR: no system dirs in {prod_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(systems)} systems\n")
    summary_rows = []

    for sys_dir in systems:
        sys_name = sys_dir.name
        is_phos = "_phos" in sys_name
        rep_dirs = sorted(sys_dir.glob("rep*"))
        print(f"=== {sys_name} ({len(rep_dirs)} reps) ===")

        for rep_dir in rep_dirs:
            print(f"  [{rep_dir.name}]")
            if not (rep_dir / "prod_fit.xtc").is_file():
                print(f"    WARNING: prod_fit.xtc missing — run postprocess first")
                continue

            ana_dir = rep_dir / "analysis"
            ana_dir.mkdir(exist_ok=True)

            final_rmsd = final_rg = mean_rmsf = mean_sasa = None

            try:
                t, rmsd = run_gmx_rms(rep_dir, gmx, ana_dir)
                final_rmsd = float(rmsd[-100:].mean()) if len(rmsd) >= 100 else float(rmsd.mean())
                print(f"    RMSD: {final_rmsd:.3f} nm")
            except Exception as e:
                print(f"    RMSD failed: {e}")

            try:
                t, rg = run_gmx_rg(rep_dir, gmx, ana_dir)
                final_rg = float(rg[-100:].mean()) if len(rg) >= 100 else float(rg.mean())
                print(f"    Rg:   {final_rg:.3f} nm")
            except Exception as e:
                print(f"    Rg failed: {e}")

            try:
                res, rmsf = run_gmx_rmsf(rep_dir, gmx, ana_dir, args.rmsf_start_ps)
                mean_rmsf = float(rmsf.mean()) if len(rmsf) > 0 else None
                print(f"    RMSF: {mean_rmsf:.3f} nm")
            except Exception as e:
                print(f"    RMSF failed: {e}")

            try:
                t, sasa = run_gmx_sasa(rep_dir, gmx, ana_dir, args.phos_resnum)
                mean_sasa = float(sasa.mean()) if len(sasa) > 0 else None
                print(f"    SASA: {mean_sasa:.3f} nm²")
            except Exception as e:
                print(f"    SASA failed: {e}")

            if is_phos:
                try:
                    result = compute_phos_contacts_from_dir(
                        rep_dir, args.phos_resnum, cutoff_A=5.0, n_sample=200
                    )
                    if result is not None:
                        residues, fracs = result
                        with open(ana_dir / "phos_contacts.csv", "w", newline="") as fh:
                            cw = csv.writer(fh)
                            cw.writerow(["residue", "contact_fraction"])
                            for r, f in zip(residues, fracs):
                                cw.writerow([int(r), f"{f:.4f}"])
                        top = sorted(zip(residues, fracs), key=lambda x: -x[1])[:5]
                        print(f"    Top contacts: " +
                              ", ".join(f"R{int(r)}={f:.2f}" for r, f in top))
                except Exception as e:
                    print(f"    Phosphate contacts failed: {e}")

            summary_rows.append({
                "system_name": sys_name,
                "is_phos": int(is_phos),
                "replicate": rep_dir.name,
                "rmsd_final_nm": f"{final_rmsd:.4f}" if final_rmsd is not None else "",
                "rg_final_nm": f"{final_rg:.4f}" if final_rg is not None else "",
                "rmsf_mean_nm": f"{mean_rmsf:.4f}" if mean_rmsf is not None else "",
                "sasa_phos_mean_nm2": f"{mean_sasa:.4f}" if mean_sasa is not None else "",
            })

        # Aggregate
        try:
            _aggregate_system(sys_dir, out_dir, parse_xvg, aggregate_replicates, write_mean_sem_csv, is_phos)
        except Exception as e:
            print(f"  Aggregation failed: {e}")

    if summary_rows:
        cols = list(summary_rows[0].keys())
        with open(out_dir / "all_systems_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary: {out_dir / 'all_systems_summary.csv'}")

    print(f"\nDone. Output in: {out_dir}")


def _aggregate_system(sys_dir, out_dir, parse_xvg, aggregate_replicates, write_mean_sem_csv, is_phos):
    """Aggregate multi-replicate CSVs for one system."""
    import csv as _csv
    import numpy as np

    rep_dirs = sorted(sys_dir.glob("rep*"))
    if not rep_dirs:
        return

    agg_dir = out_dir / sys_dir.name / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    for metric, csv_name, col_names in [
        ("rmsd", "rmsd.xvg", ("time_ns", "rmsd_mean_nm", "rmsd_sem_nm")),
        ("rg",   "rg.xvg",   ("time_ns", "rg_mean_nm", "rg_sem_nm")),
        ("sasa", "sasa.xvg", ("time_ns", "sasa_mean_nm2", "sasa_sem_nm2")),
    ]:
        series = []
        for rd in rep_dirs:
            f = rd / "analysis" / csv_name
            if f.is_file():
                t, v = parse_xvg(f)
                series.append((t, v))
        if series:
            ref_t, mean, sem = aggregate_replicates(series)
            write_mean_sem_csv(agg_dir / f"{metric}_mean_sem.csv", ref_t, mean, sem, col_names)

    # RMSF (special format)
    rmsf_data = []
    for rd in rep_dirs:
        f = rd / "analysis" / "rmsf.xvg"
        if f.is_file():
            pairs = []
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith(("#", "@")):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pairs.append((int(parts[0]), float(parts[1])))
                        except ValueError:
                            pass
            if pairs:
                rmsf_data.append(pairs)
    if rmsf_data:
        ref_res = [r for r, _ in rmsf_data[0]]
        aligned = np.array([[dict(p).get(r, np.nan) for r in ref_res] for p in rmsf_data])
        mean = np.nanmean(aligned, axis=0)
        sem = (np.nanstd(aligned, axis=0, ddof=1) / np.sqrt(len(rmsf_data))
               if len(rmsf_data) > 1 else np.zeros_like(mean))
        with open(agg_dir / "rmsf_mean_sem.csv", "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["residue", "rmsf_mean_nm", "rmsf_sem_nm"])
            for r, m, s in zip(ref_res, mean, sem):
                w.writerow([r, f"{m:.6f}", f"{s:.6f}"])

    # Phosphate contacts
    if is_phos:
        contact_data = []
        for rd in rep_dirs:
            f = rd / "analysis" / "phos_contacts.csv"
            if f.is_file():
                with open(f) as fh:
                    reader = _csv.DictReader(fh)
                    pairs = [(int(row["residue"]), float(row["contact_fraction"])) for row in reader]
                    if pairs:
                        contact_data.append(pairs)
        if contact_data:
            ref_res = [r for r, _ in contact_data[0]]
            aligned = np.array([[dict(p).get(r, 0.0) for r in ref_res] for p in contact_data])
            mean = aligned.mean(axis=0)
            sem = (aligned.std(axis=0, ddof=1) / np.sqrt(len(contact_data))
                   if len(contact_data) > 1 else np.zeros_like(mean))
            with open(agg_dir / "phos_contacts_mean_sem.csv", "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["residue", "contact_fraction_mean", "contact_fraction_sem"])
                for r, m, s in zip(ref_res, mean, sem):
                    w.writerow([r, f"{m:.4f}", f"{s:.4f}"])

    print(f"  Aggregated -> {agg_dir}")
