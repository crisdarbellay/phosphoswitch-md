#!/usr/bin/env python3
"""
analyze_trajectory.py — Per-replicate RMSD, Rg, RMSF, SASA, contacts (standalone).

Usage: python scripts/analyze_trajectory.py --prod-dir prod_outputs/ --out-dir md_analysis/
Or via CLI after pip install: analyze-trajectory --prod-dir prod_outputs/ --out-dir md_analysis/
"""
from phosphoswitch_md._cli.analyze_trajectory import main

if __name__ == "__main__":
    main()
