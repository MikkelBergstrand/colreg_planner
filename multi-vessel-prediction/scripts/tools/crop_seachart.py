"""scripts/tools/crop_seachart.py

Crop the Kartverket depth-data FGDB (all of Norway, ~4.5 GB) to a small
GeoPackage holding only the layers and area the sea-chart background needs.

Usage
-----
    python scripts/tools/crop_seachart.py                   # Moss–Horten -> handover/seacharts/
    python scripts/tools/crop_seachart.py --bbox 59.2 59.62 10.2 10.9 --out my_area.gpkg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyogrio
import shapely
from pyproj import Transformer

from colregs_planner.seachart.loader import CHART_CRS, CROPPED_CHART_NAME, FGDB_NAME, LAYER_ROLES

_ROOT = Path(__file__).resolve().parents[2]

# Paper area (59.27–59.55 N, 10.32–10.77 E) plus a margin for the plot extent.
_DEFAULT_BBOX = (59.20, 59.62, 10.20, 10.90)   # lat_min, lat_max, lon_min, lon_max


def _chart_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    """Lat/lon box -> enclosing (x_min, y_min, x_max, y_max) in the chart CRS."""
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{CHART_CRS}", always_xy=True)
    xs, ys = t.transform([lon_min, lon_max, lon_min, lon_max],
                         [lat_min, lat_min, lat_max, lat_max])
    return min(xs), min(ys), max(xs), max(ys)


def _keep_family(geoms: np.ndarray, polygonal: bool) -> np.ndarray:
    """Drop empty results and parts of the wrong dimension left by clipping."""
    dim = 2 if polygonal else 1
    parts = shapely.get_parts(geoms)
    parts = parts[~shapely.is_empty(parts) & (shapely.get_dimensions(parts) == dim)]
    return parts


def crop(src: Path, out: Path, bbox_ll: tuple[float, float, float, float]) -> None:
    bbox = _chart_bbox(*bbox_ll)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    for layer in LAYER_ROLES:
        meta, _, wkb, _ = pyogrio.raw.read(str(src), layer=layer, bbox=bbox, read_geometry=True,
                                           columns=[])
        polygonal = "Polygon" in meta["geometry_type"]
        geoms = shapely.clip_by_rect(shapely.from_wkb(wkb), *bbox) if wkb is not None else np.array([])
        parts = _keep_family(geoms, polygonal)
        pyogrio.raw.write(
            str(out), geometry=shapely.to_wkb(parts), field_data=[], fields=[],
            layer=layer, driver="GPKG", crs=f"EPSG:{CHART_CRS}",
            geometry_type="Polygon" if polygonal else "LineString",
            append=out.exists(),
        )
        print(f"  {layer:12s} {len(parts):6d} features")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=_ROOT / "seacharts" / FGDB_NAME,
                        help="Full Kartverket FGDB (default: seacharts/%(default)s)")
    parser.add_argument("--out", type=Path, default=_ROOT / "handover" / "seacharts" / CROPPED_CHART_NAME)
    parser.add_argument("--bbox", type=float, nargs=4, default=_DEFAULT_BBOX,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    args = parser.parse_args()
    crop(args.src, args.out, tuple(args.bbox))


if __name__ == "__main__":
    main()
