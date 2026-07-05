"""Console entry point: ``postprocess`` — strip waters and fit trajectories."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd=None, stdin=None, check=True):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        input=stdin, capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}):\n"
            f"  STDOUT: {result.stdout[-500:]}\n"
            f"  STDERR: {result.stderr[-500:]}"
        )
    return result


def _process_replicate(rep_dir: Path, gmx: str, keep_water: bool, skip_existing: bool) -> bool:
    prod_tpr = rep_dir / "prod.tpr"
    prod_xtc = rep_dir / "prod.xtc"

    if not prod_tpr.is_file() or not prod_xtc.is_file():
        print(f"  WARNING: {rep_dir.name}: missing prod.tpr or prod.xtc")
        return False

    fit_xtc = rep_dir / "prod_fit.xtc"
    protein_pdb = rep_dir / "prod.pdb"

    if skip_existing and fit_xtc.is_file() and protein_pdb.is_file():
        print(f"  SKIP: {rep_dir.name}: outputs exist")
        return True

    sel = "0" if keep_water else "1"

    whole_xtc = rep_dir / "prod_whole.xtc"
    if not whole_xtc.is_file():
        _run([gmx, "trjconv", "-s", "prod.tpr", "-f", "prod.xtc",
              "-o", "prod_whole.xtc", "-pbc", "whole", "-ur", "compact"],
             cwd=rep_dir, stdin=f"{sel}\n")

    nojump_xtc = rep_dir / "prod_nojump.xtc"
    if not nojump_xtc.is_file():
        _run([gmx, "trjconv", "-s", "prod.tpr", "-f", "prod_whole.xtc",
              "-o", "prod_nojump.xtc", "-pbc", "nojump"],
             cwd=rep_dir, stdin=f"{sel}\n")

    _run([gmx, "trjconv", "-s", "prod.tpr", "-f", "prod_nojump.xtc",
          "-o", "prod_fit.xtc", "-fit", "rot+trans"],
         cwd=rep_dir, stdin=f"3\n{sel}\n")

    if not protein_pdb.is_file():
        _run([gmx, "trjconv", "-s", "prod.tpr", "-f", "prod_fit.xtc",
              "-o", "prod.pdb", "-dump", "0"],
             cwd=rep_dir, stdin=f"{sel}\n")

    for tmp in [whole_xtc, nojump_xtc]:
        if tmp.is_file():
            tmp.unlink()

    return True


def main(argv=None):
    """Strip waters, re-image PBC, and fit Cα for all replicate trajectories."""
    ap = argparse.ArgumentParser(
        prog="postprocess",
        description="Strip waters and Cα-fit trajectories after cluster download.",
    )
    ap.add_argument("--prod-dir", required=True,
                    help="Root directory with <system>/<rep>/prod.tpr + prod.xtc")
    ap.add_argument("--gmx", default="gmx", help="gmx binary (default: gmx in PATH)")
    ap.add_argument("--keep-water", action="store_true",
                    help="Keep solvent in output (default: strip)")
    ap.add_argument("--no-skip", action="store_true",
                    help="Reprocess even if outputs already exist")
    args = ap.parse_args(argv)

    prod_dir = Path(args.prod_dir).resolve()
    if not prod_dir.is_dir():
        print(f"ERROR: not a directory: {prod_dir}", file=sys.stderr)
        sys.exit(1)

    gmx_bin = shutil.which(args.gmx) or args.gmx
    print(f"Using gmx: {gmx_bin}\n")

    replicates = []
    for sys_dir in sorted(prod_dir.iterdir()):
        if sys_dir.is_dir():
            for rep_dir in sorted(sys_dir.glob("rep*")):
                if rep_dir.is_dir():
                    replicates.append(rep_dir)

    if not replicates:
        print(f"ERROR: no rep*/ subdirs in {prod_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(replicates)} replicates\n")
    n_ok = n_fail = 0
    for i, rd in enumerate(replicates, 1):
        print(f"[{i}/{len(replicates)}] {rd.parent.name}/{rd.name}")
        try:
            ok = _process_replicate(rd, gmx_bin, args.keep_water, not args.no_skip)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            n_fail += 1

    print(f"\n{'='*60}\n  Processed: {n_ok}/{len(replicates)}")
    if n_fail:
        print(f"  Failed:    {n_fail}")
    print(f"{'='*60}")
