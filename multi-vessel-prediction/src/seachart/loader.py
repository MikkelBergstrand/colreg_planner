"""seachart/loader.py

Loads sea chart geometry from a Kartverket FGDB file and returns it in the
local NED (North-East-Down) coordinate frame so it can be drawn on a
matplotlib axis calibrated in metres.

The FGDB uses EPSG:25833 (UTM Zone 33N).  Conversion chain:
    NED [m] → WGS84 lat/lon → UTM 25833 (for spatial query) → back to NED

Dependencies (optional — module degrades gracefully if absent):
    pyogrio  : fast GDAL-based FGDB reader with proper spatial filtering
    pyproj   : CRS transform EPSG:25833 <-> EPSG:4326
    shapely  : WKB geometry parsing

Run scripts/explore/inspect_seachart.py once to discover the actual layer names in
your FGDB, then update LAYER_ROLES below.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

FGDB_NAME          = "Basisdata_0000_Norge_25833_Dybdedata_FGDB.gdb"   # full chart, all of Norway
CROPPED_CHART_NAME = "moss_horten.gpkg"   # Moss–Horten crop, scripts/tools/crop_seachart.py
CHART_CRS          = 25833                # EPSG code of both files
_REPO_ROOT = Path(__file__).resolve().parents[2]


def find_chart() -> Path | None:
    """The sea chart in seacharts/ or handover/seacharts/, or None.

    Prefers the full FGDB, which covers every area, over the Moss–Horten crop.
    """
    for name in (FGDB_NAME, CROPPED_CHART_NAME):
        for folder in ("seacharts", "handover/seacharts"):
            path = _REPO_ROOT / folder / name
            if path.exists():
                return path
    return None


try:
    import pyogrio
    _PYOGRIO_OK = True
except ImportError:
    _PYOGRIO_OK = False

try:
    from pyproj import Transformer
    _PYPROJ_OK = True
except ImportError:
    _PYPROJ_OK = False

try:
    from shapely import from_wkb, get_parts
    from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
    _SHAPELY_OK = True
except ImportError:
    _SHAPELY_OK = False


# ---------------------------------------------------------------------------
# Layer → role mapping
# ---------------------------------------------------------------------------
# Keys are the exact FGDB layer names (case-sensitive Norwegian/SOSI names).
# Run scripts/explore/inspect_seachart.py to verify these against your FGDB.

LAYER_ROLES: dict[str, str] = {
    "landareal":  "land",           # land areas (MultiPolygon)
    "kystkontur": "coastline",      # coastline (MultiLineString)
    "dybdekurve": "depth_contour",  # depth contours / isobaths (MultiLineString)
    "torrfall":   "shoal",          # drying/tidal areas — dry at low tide (MultiPolygon)
}

_LAT_TO_M = 111320.0  # metres per degree latitude (same constant as analyze_test_data.py)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ChartBackground:
    """Sea chart geometry for one viewport, pre-converted to NED metres.

    Each list holds numpy arrays — one array per feature ring/segment.
    Polygon arrays: shape (N, 2), columns [east_m, north_m].
    Line arrays:    shape (M, 2), columns [east_m, north_m].
    """
    land_polys:      list[np.ndarray] = field(default_factory=list)
    shoal_polys:     list[np.ndarray] = field(default_factory=list)
    coastline_segs:  list[np.ndarray] = field(default_factory=list)
    depth_contours:  list[np.ndarray] = field(default_factory=list)
    ref_lat:         float = 0.0
    ref_lon:         float = 0.0
    layer_counts:    dict[str, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            len(self.land_polys)     == 0
            and len(self.shoal_polys)    == 0
            and len(self.coastline_segs) == 0
            and len(self.depth_contours) == 0
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_land_polygons(
    fgdb_path: str | Path,
    bbox: tuple[float, float, float, float],
    source_crs: int = 32632,
    target_crs: int = 32632,
):
    """Load land-area polygons from the FGDB within a bounding box.

    Parameters
    ----------
    fgdb_path  : path to the .gdb directory
    bbox       : (x_min, y_min, x_max, y_max) in *source_crs*
    source_crs : EPSG code of the bbox coordinates (default: 32632 = UTM 32N)
    target_crs : EPSG code for returned polygon coordinates (default: 32632)

    Returns
    -------
    shapely.geometry.MultiPolygon, or None if no land features found.
    """
    if not (_PYOGRIO_OK and _PYPROJ_OK and _SHAPELY_OK):
        warnings.warn(
            "pyogrio / pyproj / shapely required for load_land_polygons().",
            RuntimeWarning, stacklevel=2,
        )
        return None

    from shapely.ops import unary_union

    # Transform bbox to FGDB native CRS (25833) for spatial query
    to_fgdb = Transformer.from_crs(source_crs, 25833, always_xy=True)
    x_min, y_min = to_fgdb.transform(bbox[0], bbox[1])
    x_max, y_max = to_fgdb.transform(bbox[2], bbox[3])
    fgdb_bbox = (x_min, y_min, x_max, y_max)

    available = {str(n) for n, _ in pyogrio.list_layers(str(fgdb_path))}
    if "landareal" not in available:
        warnings.warn("Layer 'landareal' not found in FGDB.", RuntimeWarning, stacklevel=2)
        return None

    try:
        _, _, geom_wkb, _ = pyogrio.raw.read(
            str(fgdb_path), layer="landareal", bbox=fgdb_bbox,
        )
    except Exception as exc:
        warnings.warn(f"Error reading landareal: {exc}", RuntimeWarning, stacklevel=2)
        return None

    if geom_wkb is None or len(geom_wkb) == 0:
        return None

    # Transform coordinates from FGDB CRS to target CRS
    to_target = Transformer.from_crs(25833, target_crs, always_xy=True)

    def _transform_polygon(poly):
        coords = np.array(poly.exterior.coords)
        xs, ys = to_target.transform(coords[:, 0], coords[:, 1])
        return Polygon(np.column_stack([xs, ys]))

    polygons = []
    for geom in from_wkb(geom_wkb):
        if geom is None or geom.is_empty:
            continue
        parts = get_parts(geom) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            if part is not None and not part.is_empty:
                polygons.append(_transform_polygon(part))

    if not polygons:
        return None

    result = unary_union(polygons)
    if result.geom_type == "Polygon":
        return MultiPolygon([result])
    return result


def list_layers(fgdb_path: str | Path) -> list[str]:
    """Return all layer names in the FGDB, sorted alphabetically."""
    if not _PYOGRIO_OK:
        warnings.warn("pyogrio not installed — cannot list layers.", RuntimeWarning, stacklevel=2)
        return []
    layers = pyogrio.list_layers(str(fgdb_path))
    return sorted(str(name) for name, _ in layers)


def load_chart_background(
    fgdb_path: str | Path,
    ref_lat: float,
    ref_lon: float,
    ned_xlim: tuple[float, float],
    ned_ylim: tuple[float, float],
    layer_roles: Optional[dict[str, str]] = None,
) -> ChartBackground:
    """Load sea chart features for a NED viewport and return geometry in NED metres.

    Parameters
    ----------
    fgdb_path   : path to the .gdb directory
    ref_lat     : WGS84 latitude of the NED origin [deg]
    ref_lon     : WGS84 longitude of the NED origin [deg]
    ned_xlim    : (e_min, e_max) east bounds [m]
    ned_ylim    : (n_min, n_max) north bounds [m]
    layer_roles : override for LAYER_ROLES; if None the module-level dict is used

    Returns
    -------
    ChartBackground
        Populated dataclass, or empty ChartBackground if deps are missing.
    """
    bg = ChartBackground(ref_lat=ref_lat, ref_lon=ref_lon)

    for dep, ok, name in [
        (_PYOGRIO_OK,  True, "pyogrio"),
        (_PYPROJ_OK,   True, "pyproj"),
        (_SHAPELY_OK,  True, "shapely"),
    ]:
        if not dep:
            warnings.warn(
                f"{name} not installed — chart background empty.\n"
                "Install with: pip install pyogrio pyproj shapely",
                RuntimeWarning, stacklevel=2,
            )
            return bg

    roles = layer_roles if layer_roles is not None else LAYER_ROLES
    available = {str(n) for n, _ in pyogrio.list_layers(str(fgdb_path))}

    try:
        utm_bbox = _ned_bbox_to_utm(ref_lat, ref_lon, ned_xlim, ned_ylim)
    except Exception as exc:
        warnings.warn(f"Could not compute UTM bbox: {exc}", RuntimeWarning, stacklevel=2)
        return bg

    inv_tf = Transformer.from_crs(25833, 4326, always_xy=True)
    cos_lat = math.cos(math.radians(ref_lat))

    _role_dest: dict[str, list] = {
        "land":          bg.land_polys,
        "shoal":         bg.shoal_polys,
        "coastline":     bg.coastline_segs,
        "depth_contour": bg.depth_contours,
    }

    for layer_name, role in roles.items():
        if layer_name not in available:
            warnings.warn(
                f"FGDB layer '{layer_name}' not found — skipping. "
                "Run scripts/explore/inspect_seachart.py to see available names.",
                RuntimeWarning, stacklevel=2,
            )
            continue

        dest = _role_dest.get(role)
        if dest is None:
            warnings.warn(f"Unknown role '{role}' for layer '{layer_name}' — skipping.")
            continue

        is_polygon = role in ("land", "shoal")
        try:
            _, _, geom_wkb, _ = pyogrio.raw.read(
                str(fgdb_path), layer=layer_name, bbox=utm_bbox,
            )
            geoms = from_wkb(geom_wkb)
            count = 0
            for geom in geoms:
                if geom is None:
                    continue
                count += 1
                if is_polygon:
                    dest.extend(_polygon_to_ned(geom, ref_lat, ref_lon, cos_lat, inv_tf))
                else:
                    dest.extend(_line_to_ned(geom, ref_lat, ref_lon, cos_lat, inv_tf))
            bg.layer_counts[role] = count
        except Exception as exc:
            warnings.warn(
                f"Error reading layer '{layer_name}': {exc}", RuntimeWarning, stacklevel=2,
            )

    return bg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ned_bbox_to_utm(
    ref_lat: float,
    ref_lon: float,
    ned_xlim: tuple[float, float],
    ned_ylim: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Convert NED viewport bounds to a UTM 25833 bounding box for pyogrio queries."""
    cos_lat = math.cos(math.radians(ref_lat))
    lat_min = ref_lat + ned_ylim[0] / _LAT_TO_M
    lat_max = ref_lat + ned_ylim[1] / _LAT_TO_M
    lon_min = ref_lon + ned_xlim[0] / (_LAT_TO_M * cos_lat)
    lon_max = ref_lon + ned_xlim[1] / (_LAT_TO_M * cos_lat)

    tf = Transformer.from_crs(4326, 25833, always_xy=True)
    x_min, y_min = tf.transform(lon_min, lat_min)
    x_max, y_max = tf.transform(lon_max, lat_max)
    return x_min, y_min, x_max, y_max


def _coords_to_ned(
    coords: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    cos_lat: float,
    inv_tf: "Transformer",
) -> np.ndarray:
    """Convert an (N, 2) array of UTM 25833 coords to NED [east, north] metres."""
    lons, lats = inv_tf.transform(coords[:, 0], coords[:, 1])
    east  = (np.asarray(lons) - ref_lon) * _LAT_TO_M * cos_lat
    north = (np.asarray(lats) - ref_lat) * _LAT_TO_M
    return np.column_stack([east, north])


def _polygon_to_ned(
    geom,
    ref_lat: float,
    ref_lon: float,
    cos_lat: float,
    inv_tf: "Transformer",
) -> list[np.ndarray]:
    """Extract exterior rings from a Polygon/MultiPolygon as NED arrays."""
    parts = get_parts(geom) if geom.geom_type == "MultiPolygon" else [geom]
    result = []
    for poly in parts:
        if poly is None or poly.is_empty:
            continue
        coords = np.array(poly.exterior.coords)
        if len(coords) >= 3:
            result.append(_coords_to_ned(coords, ref_lat, ref_lon, cos_lat, inv_tf))
    return result


def _line_to_ned(
    geom,
    ref_lat: float,
    ref_lon: float,
    cos_lat: float,
    inv_tf: "Transformer",
) -> list[np.ndarray]:
    """Extract coordinate arrays from a LineString/MultiLineString as NED arrays."""
    parts = get_parts(geom) if geom.geom_type == "MultiLineString" else [geom]
    result = []
    for line in parts:
        if line is None or line.is_empty:
            continue
        coords = np.array(line.coords)
        if len(coords) >= 2:
            result.append(_coords_to_ned(coords, ref_lat, ref_lon, cos_lat, inv_tf))
    return result
