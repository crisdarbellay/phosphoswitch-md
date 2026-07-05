#!/usr/bin/env python3
"""
basin_occupancy.py — Dual-reference basin occupancy analysis (standalone).

Usage: python scripts/basin_occupancy.py --prod-dir ... --af3-dir ... --out-dir ...
Or via CLI after pip install: basin-occupancy --prod-dir ... --af3-dir ... --out-dir ...
"""
from phosphoswitch_md._cli.basin_occupancy import main

if __name__ == "__main__":
    main()
