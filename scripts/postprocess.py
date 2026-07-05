#!/usr/bin/env python3
"""
postprocess.py — Strip waters and fit Cα trajectories (standalone script).

Usage: python scripts/postprocess.py --prod-dir prod_outputs/ --gmx gmx
Or via CLI after pip install: postprocess --prod-dir prod_outputs/ --gmx gmx
"""
from phosphoswitch_md._cli.postprocess import main

if __name__ == "__main__":
    main()
