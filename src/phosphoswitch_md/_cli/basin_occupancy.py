"""Console entry point: ``basin-occupancy``."""
import argparse
import csv
import sys
from pathlib import Path


def main(argv=None):
    """Dual-reference basin occupancy analysis for all phosphoswitch candidates."""
    ap = argparse.ArgumentParser(
        prog="basin-occupancy",
        description="Compute dual-reference basin occupancy and switch metrics.",
    )
    ap.add_argument("--prod-dir", required=True)
    ap.add_argument("--af3-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--strict-threshold", type=float, default=4.0)
    ap.add_argument("--loose-threshold", type=float, default=7.0)
    ap.add_argument("--n-frames", type=int, default=1000)
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--phos-resnum", type=int, default=30)
    args = ap.parse_args(argv)

    try:
        from phosphoswitch_md.io_utils import find_candidates
        from phosphoswitch_md.basin import analyze_candidate_basin_occupancy
        from phosphoswitch_md.figures import (
            plot_occupancy_bars, plot_landscape_2d, plot_min_rmsd_timeseries,
        )
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    prod_dir = Path(args.prod_dir).resolve()
    af3_dir = Path(args.af3_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    strict = args.strict_threshold
    loose = args.loose_threshold

    candidates = find_candidates(prod_dir)
    if args.candidates:
        candidates = [c for c in candidates if c in args.candidates]

    print(f"Found {len(candidates)} candidates. strict={strict}Å, loose={loose}Å\n")

    per_rep_rows = []
    switch_rows = []

    for ci, cand in enumerate(candidates, 1):
        print(f"\n[{ci}/{len(candidates)}] {cand}")
        ref_apo = af3_dir / cand / "apo.pdb"
        ref_phos = af3_dir / cand / "phos.pdb"
        if not ref_apo.exists() or not ref_phos.exists():
            print(f"  WARNING: AF3 references missing — skipping")
            continue

        cand_dir = out_dir / cand
        cand_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = analyze_candidate_basin_occupancy(
                cand, prod_dir, ref_apo, ref_phos,
                n_sample=args.n_frames, strict_A=strict, loose_A=loose,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        ref_rmsd = results.get("ref_rmsd_A", float("nan"))
        print(f"  AF3 apo-phos RMSD = {ref_rmsd:.2f} Å")

        for cond in ("apo", "phos"):
            for rd in results.get(cond, {}).get("reps", []):
                m = rd["metrics"]
                print(f"    {cond}/{rd['rep']}: "
                      f"min_phos={m['min_rmsd_to_phos']:.1f}Å "
                      f"frac_phos_loose={m['frac_phos_loose']:.3f}")
                per_rep_rows.append({
                    "candidate": cand, "condition": cond, "rep": rd["rep"],
                    "ref_apo_phos_rmsd_A": round(ref_rmsd, 2),
                    **{k: round(v, 4) for k, v in m.items() if isinstance(v, float)},
                    "hitting_time_strict_ns": rd["hitting_time_strict_ns"],
                    "hitting_time_loose_ns": rd["hitting_time_loose_ns"],
                })

        agg_apo = results["apo"].get("aggregated", {})
        agg_phos = results["phos"].get("aggregated", {})

        if agg_apo and agg_phos:
            try:
                plot_occupancy_bars(agg_apo, agg_phos, cand, cand_dir, strict, loose)
                print(f"    -> {cand}_occupancy_bars.png")
            except Exception as e:
                print(f"    occupancy bars failed: {e}")

        cand_landscape = {
            cond: [{"rmsd_apo": r["rmsd_to_apo"], "rmsd_phos": r["rmsd_to_phos"]}
                   for r in results.get(cond, {}).get("reps", [])]
            for cond in ("apo", "phos")
        }
        cand_ts = {
            cond: [{"time_ns": r["time_ns"], "rmsd_apo": r["rmsd_to_apo"],
                    "rmsd_phos": r["rmsd_to_phos"], "rep": r["rep"]}
                   for r in results.get(cond, {}).get("reps", [])]
            for cond in ("apo", "phos")
        }

        if any(cand_landscape.values()):
            try:
                plot_landscape_2d(cand_landscape, cand, cand_dir, strict, loose,
                                  x_max=max(30.0, ref_rmsd + 10),
                                  y_max=max(30.0, ref_rmsd + 10))
                print(f"    -> {cand}_landscape_2d.png")
            except Exception as e:
                print(f"    landscape failed: {e}")

        if any(cand_ts.values()):
            try:
                plot_min_rmsd_timeseries(cand_ts, cand, cand_dir)
                print(f"    -> {cand}_min_rmsd_timeseries.png")
            except Exception as e:
                print(f"    timeseries failed: {e}")

        sm = results.get("switch_metrics", {})
        if sm and agg_apo and agg_phos:
            switch_rows.append({
                "candidate": cand,
                "ref_apo_phos_rmsd_A": round(ref_rmsd, 2),
                "apo_MD_frac_phos_loose": round(agg_apo.get("frac_phos_loose", float("nan")), 4),
                "phos_MD_frac_phos_loose": round(agg_phos.get("frac_phos_loose", float("nan")), 4),
                **{k: v for k, v in sm.items()},
            })

    # Write outputs
    def _write(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write(out_dir / "per_rep_basin_metrics.csv", per_rep_rows)
    switch_rows.sort(key=lambda r: -(r.get("enrich_phos_loose") or 0))
    _write(out_dir / "switch_metrics.csv", switch_rows)

    print(f"\nOutputs in: {out_dir}")

    # Scorecard
    print("\n" + "=" * 80)
    print("SWITCH SCORECARD")
    print("=" * 80)
    print(f"{'Candidate':<40} {'apoMD%':>8} {'phosMD%':>8} {'enrich':>8} {'log2':>7}")
    for row in switch_rows:
        print(f"{row['candidate']:<40} "
              f"{row['apo_MD_frac_phos_loose'] * 100:>7.1f}% "
              f"{row['phos_MD_frac_phos_loose'] * 100:>7.1f}% "
              f"{row['enrich_phos_loose']:>8.2f} "
              f"{row.get('log2_phos_enrich_loose', float('nan')):>+7.2f}")
    print("=" * 80)
