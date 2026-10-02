"""scripts/explore/inspect_seachart.py

One-shot script to list all layer names, geometry types, and feature counts
inside the Kartverket depth-data FGDB.

Usage
-----
    python scripts/explore/inspect_seachart.py

After running, update LAYER_ROLES in src/seachart/loader.py with the actual
layer names that correspond to land, coastline, depth contours, and shoals.
"""
from __future__ import annotations

import sys
from pathlib import Path

GDB_PATH = Path(__file__).resolve().parents[2] / "seacharts" / \
           "Basisdata_0000_Norge_25833_Dybdedata_FGDB.gdb"

try:
    import pyogrio
except ImportError:
    print("ERROR: pyogrio not installed.  Run: pip install pyogrio pyproj shapely")
    sys.exit(1)

if not GDB_PATH.exists():
    print(f"ERROR: FGDB not found at {GDB_PATH}")
    sys.exit(1)

layers = pyogrio.list_layers(str(GDB_PATH))
print(f"\nFGDB: {GDB_PATH.name}")
print(f"Total layers: {len(layers)}\n")
print(f"{'Layer':<45} {'GeomType':<22} {'Count':>8}")
print("-" * 78)
for name, geom_type in sorted(layers, key=lambda x: x[0]):
    try:
        info = pyogrio.read_info(str(GDB_PATH), layer=str(name))
        count = info["features"]
    except Exception as exc:
        count = f"ERROR: {exc}"
    print(f"{str(name):<45} {str(geom_type):<22} {str(count):>8}")
