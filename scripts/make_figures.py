#!/usr/bin/env python3
"""
make_figures.py — Generate all paper figures from aggregated CSVs (standalone).

Usage: python scripts/make_figures.py --analysis-dir md_analysis/ --out-dir md_figures/
Or via CLI after pip install: make-figures --analysis-dir md_analysis/ --out-dir md_figures/
"""
from phosphoswitch_md._cli.make_figures import main

if __name__ == "__main__":
    main()
