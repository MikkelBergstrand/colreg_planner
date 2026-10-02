"""scripts/tools/analyze_deviation.py

Split per_sample rows from one or more ``metrics_win*.json`` files (as written
by ``run_AIS.py``'s metrics pipeline) into two groups and compare ADE/FDE
between them, for the interaction and nominal methods. Three modes are
available via --split-by:

  deviation (default) — "deviating" (dev_m > D) vs "non-deviating" (dev_m <= D),
      where dev_m is how far the interaction-aware prediction departs from the
      nominal-only one for the same (window, frame, vessel). D defaults to
      d_pass read from the metrics file's own config.

  role — "burden-role" vs "no-burden-role", where has_burden_role marks a
      vessel that held a non-STAND-ON COLREGs role (GIVE_WAY / HEAD_ON /
      OVERTAKING / AVOID_COLLISION) toward at least one counterpart that
      frame, per the raw classifier graph.

  unconditional — no split: one "Table 1"-style summary, mean ADE/FDE per
      method (interaction / nominal / constant_velocity) across the horizons
      in --horizons (not just interaction/nominal — the full ablation).

Both dev_m and ade_m/fde_m are only meaningful per horizon (dev_m is a max
over that horizon's rollout; ade_m/fde_m are averaged over it), so deviation/
role selects a single horizon via --horizon-s and uses it throughout — never
mixed across horizons. has_burden_role isn't horizon-dependent itself (it's a
per-frame encounter property) but rows are still read at the selected horizon
so the ADE/FDE side of the table stays consistent with the deviation split.
unconditional instead reports multiple horizons side by side (--horizons),
each still computed from its own, independently-pooled rows — never mixing
one horizon's n into another's mean.

Usage
-----
    python scripts/tools/analyze_deviation.py outputs/data/metrics_win054_*.json
    python scripts/tools/analyze_deviation.py outputs/data/metrics_win*.json --horizon-s 600
    python scripts/tools/analyze_deviation.py outputs/data/metrics_win*.json --threshold 150
    python scripts/tools/analyze_deviation.py outputs/data/metrics_win*.json --split-by role
    python scripts/tools/analyze_deviation.py outputs/data/metrics_win*.json --split-by unconditional
    python scripts/tools/analyze_deviation.py outputs/data/metrics_win*.json --split-by unconditional --horizons 60,300,600
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

Key = tuple  # (file_index, window_id, frame, vessel_id)

_TABLE_METHODS = ("interaction", "nominal")
_ALL_METHODS   = ("interaction", "nominal", "constant_velocity")

_DEFAULT_HORIZON_S = 600.0
_DEFAULT_HORIZONS  = "60,300,600"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(paths: list[Path]) -> list[tuple[Path, dict]]:
    docs = []
    for p in paths:
        with open(p) as f:
            docs.append((p, json.load(f)))
    return docs


def _extract_d_pass(doc: dict) -> float | None:
    try:
        return float(doc["metadata"]["prediction_config"]["interaction"]["agent"]["d_pass"])
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_threshold(
    explicit: float | None, docs: list[tuple[Path, dict]],
) -> tuple[float, str]:
    if explicit is not None:
        return explicit, "explicit --threshold"

    found: list[tuple[Path, float]] = []
    for path, doc in docs:
        d = _extract_d_pass(doc)
        if d is not None:
            found.append((path, d))

    if not found:
        return 100.0, "hardcoded default (no d_pass found in any file's metadata)"

    d_pass = found[0][1]
    distinct = sorted({d for _, d in found})
    if len(distinct) > 1:
        detail = ", ".join(f"{p.name}={d}" for p, d in found)
        print(f"  Warning: files disagree on d_pass ({detail}); "
              f"using {d_pass} from {found[0][0].name}.", file=sys.stderr)
    return d_pass, f"d_pass from metadata.prediction_config ({found[0][0].name})"


def _rows_at_horizon(
    docs: list[tuple[Path, dict]], horizon_s: float, tol: float = 1e-3,
) -> tuple[dict[str, dict[Key, dict]], dict[Key, bool], list[float]]:
    """Collect per_sample rows at ``horizon_s``, keyed by
    (file_index, window_id, frame, vessel_id) so windows can't collide across
    pooled files.

    Returns
    -------
    rows_by_method : dict[method -> dict[key -> row]]  (every method present)
    burden_map     : dict[key -> bool]  has_burden_role, taken from whichever
        method's row carries it first (it's a per-frame/vessel encounter
        property, not a per-method one, so it's the same across methods).
    seen_horizons  : distinct horizon_s values found in the loaded files, for
        a helpful error message if none match ``horizon_s``.
    """
    rows_by_method: dict[str, dict[Key, dict]] = {}
    burden_map: dict[Key, bool] = {}
    seen_horizons: set[float] = set()

    for file_idx, (_path, doc) in enumerate(docs):
        for row in doc.get("per_sample", []):
            seen_horizons.add(row["horizon_s"])
            if abs(row["horizon_s"] - horizon_s) > tol:
                continue
            method = row.get("method")
            key = (file_idx, row["window_id"], row["frame"], row["vessel_id"])
            rows_by_method.setdefault(method, {})[key] = row
            if "has_burden_role" in row and key not in burden_map:
                burden_map[key] = row["has_burden_role"]

    return rows_by_method, burden_map, sorted(seen_horizons)


def _agg(vals: list[float]) -> tuple[float, float] | tuple[None, None]:
    if not vals:
        return None, None
    return statistics.mean(vals), statistics.median(vals)


# ---------------------------------------------------------------------------
# Split criteria — each resolves to two named groups of row keys
# ---------------------------------------------------------------------------

def _split_by_deviation(
    rows_by_method: dict[str, dict[Key, dict]],
    args: argparse.Namespace,
    docs: list[tuple[Path, dict]],
) -> tuple[list[tuple[str, set[Key]]], list[str]] | None:
    interaction_rows = rows_by_method.get("interaction", {})
    dev_map = {k: row["dev_m"] for k, row in interaction_rows.items() if "dev_m" in row}
    if not dev_map:
        print(f"No interaction rows with dev_m found at horizon_s={args.horizon_s}.")
        return None

    threshold, threshold_source = _resolve_threshold(args.threshold, docs)
    deviating     = {k for k, v in dev_map.items() if v >  threshold}
    non_deviating = {k for k, v in dev_map.items() if v <= threshold}
    n = len(dev_map)

    header = [
        f"Threshold D: {threshold} m  (source: {threshold_source})",
        f"Classified samples (interaction rows with dev_m @ this horizon): {n}",
        f"  deviating     (dev_m > D):  {len(deviating):5d}  ({len(deviating) / n:.1%})",
        f"  non-deviating (dev_m <= D): {len(non_deviating):5d}  ({len(non_deviating) / n:.1%})",
    ]
    return [("deviating", deviating), ("non-deviating", non_deviating)], header


def _split_by_role(
    rows_by_method: dict[str, dict[Key, dict]],
    burden_map: dict[Key, bool],
    args: argparse.Namespace,
) -> tuple[list[tuple[str, set[Key]]], list[str]] | None:
    if not burden_map:
        print(f"No rows with has_burden_role found at horizon_s={args.horizon_s} "
              f"(metrics file predates the role field — re-run the metrics pipeline).")
        return None

    burden    = {k for k, v in burden_map.items() if v}
    no_burden = {k for k, v in burden_map.items() if not v}
    n = len(burden_map)

    header = [
        f"Classified samples (rows with has_burden_role @ this horizon): {n}",
        f"  burden-role    (has_burden_role):     {len(burden):5d}  ({len(burden) / n:.1%})",
        f"  no-burden-role (not has_burden_role): {len(no_burden):5d}  ({len(no_burden) / n:.1%})",
    ]
    return [("burden-role", burden), ("no-burden-role", no_burden)], header


# ---------------------------------------------------------------------------
# Unconditional (no split) multi-horizon summary — "Table 1"
# ---------------------------------------------------------------------------

def _parse_horizons(raw: str) -> list[float]:
    try:
        return [float(h.strip()) for h in raw.split(",") if h.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --horizons value: {raw!r} "
            f"(expected comma-separated numbers, e.g. '60,300,600')")


def _hfmt(horizon_s: float) -> str:
    return f"{horizon_s:g}s"


def _unconditional_stats(
    docs: list[tuple[Path, dict]], horizons: list[float],
) -> dict[float, dict[str, tuple[int, float | None, float | None]]]:
    """For each horizon, pool per_sample rows across all loaded files
    (independently per horizon — never mixing one horizon's rows into
    another's) and compute (n, mean_ade_m, mean_fde_m) per method in
    ``_ALL_METHODS``.
    """
    stats: dict[float, dict[str, tuple[int, float | None, float | None]]] = {}
    for h in horizons:
        rows_by_method, _burden_map, _seen = _rows_at_horizon(docs, h)
        per_method: dict[str, tuple[int, float | None, float | None]] = {}
        for method in _ALL_METHODS:
            rows = rows_by_method.get(method, {})
            mean_ade, _ = _agg([r["ade_m"] for r in rows.values()])
            mean_fde, _ = _agg([r["fde_m"] for r in rows.values()])
            per_method[method] = (len(rows), mean_ade, mean_fde)
        stats[h] = per_method
    return stats


def _print_unconditional_table(
    horizons: list[float],
    stats: dict[float, dict[str, tuple[int, float | None, float | None]]],
) -> None:
    def _fmt(v: float | None, width: int) -> str:
        return f"{v:{width}.2f}" if v is not None else f"{'n/a':>{width}}"

    # n is only meaningful as one shared column if it's actually the same
    # value at every horizon, for every method (require_full_horizon can drop
    # a different number of rows at each horizon near the tail of a scored
    # window) — otherwise each horizon gets its own n right next to its own
    # ADE/FDE instead of one column claiming a single count for all of them.
    shared_n = all(
        len({stats[h][method][0] for h in horizons}) <= 1
        for method in _ALL_METHODS
    )

    if shared_n:
        header = f"{'method':<18} {'n':>6}"
        for h in horizons:
            header += f"  {'ADE@' + _hfmt(h):>10}"
        for h in horizons:
            header += f"  {'FDE@' + _hfmt(h):>10}"
        print(header)
        print("-" * len(header))
        for method in _ALL_METHODS:
            n = stats[horizons[0]][method][0] if horizons else 0
            line = f"{method:<18} {n:>6}"
            for h in horizons:
                line += f"  {_fmt(stats[h][method][1], 10)}"
            for h in horizons:
                line += f"  {_fmt(stats[h][method][2], 10)}"
            print(line)
    else:
        header = f"{'method':<18}"
        for h in horizons:
            header += f"  {'n@' + _hfmt(h):>8} {'ADE@' + _hfmt(h):>10} {'FDE@' + _hfmt(h):>10}"
        print(header)
        print("-" * len(header))
        for method in _ALL_METHODS:
            line = f"{method:<18}"
            for h in horizons:
                n, mean_ade, mean_fde = stats[h][method]
                line += f"  {n:>8} {_fmt(mean_ade, 10)} {_fmt(mean_fde, 10)}"
            print(line)


# ---------------------------------------------------------------------------
# Shared table
# ---------------------------------------------------------------------------

def _print_table(
    groups: list[tuple[str, set[Key]]],
    rows_by_method: dict[str, dict[Key, dict]],
) -> None:
    methods = [(m, rows_by_method.get(m, {})) for m in _TABLE_METHODS]

    header = (f"{'group':<16} {'method':<12} {'n':>6} "
              f"{'mean_ade_m':>11} {'median_ade_m':>13} "
              f"{'mean_fde_m':>11} {'median_fde_m':>13}")
    print(header)
    print("-" * len(header))
    for group_name, group_keys in groups:
        for method_name, method_rows in methods:
            ade_vals = [method_rows[k]["ade_m"] for k in group_keys if k in method_rows]
            fde_vals = [method_rows[k]["fde_m"] for k in group_keys if k in method_rows]
            n = len(ade_vals)
            mean_ade, med_ade = _agg(ade_vals)
            mean_fde, med_fde = _agg(fde_vals)
            fmt   = lambda v: f"{v:11.2f}" if v is not None else f"{'n/a':>11}"
            fmt13 = lambda v: f"{v:13.2f}" if v is not None else f"{'n/a':>13}"
            print(f"{group_name:<16} {method_name:<12} {n:>6} "
                  f"{fmt(mean_ade)} {fmt13(med_ade)} "
                  f"{fmt(mean_fde)} {fmt13(med_fde)}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics_files", nargs="+", type=Path,
                     help="one or more metrics_win*.json files (pooled together)")
    ap.add_argument("--horizon-s", type=float, default=_DEFAULT_HORIZON_S,
                     help="[--split-by deviation/role only] horizon (seconds) to select "
                          "rows for — dev_m, ade_m and fde_m are all taken from this "
                          "horizon only (default: 600)")
    ap.add_argument("--horizons", type=str, default=_DEFAULT_HORIZONS,
                     help="[--split-by unconditional only] comma-separated horizons in "
                          "seconds to report side by side, e.g. '60,300,600' "
                          "(default: 60,300,600)")
    ap.add_argument("--split-by", choices=["deviation", "role", "unconditional"],
                     default="deviation",
                     help="grouping criterion: 'deviation' (dev_m > threshold, default), "
                          "'role' (has_burden_role), or 'unconditional' (no split — one "
                          "multi-horizon ADE/FDE table across interaction/nominal/"
                          "constant_velocity)")
    ap.add_argument("--threshold", "-D", type=float, default=None,
                     help="[--split-by deviation only] deviation threshold in meters "
                          "(default: d_pass read from the first file's "
                          "metadata.prediction_config, else 100.0)")
    args = ap.parse_args()

    missing = [p for p in args.metrics_files if not p.exists()]
    if missing:
        ap.error(f"file(s) not found: {', '.join(str(p) for p in missing)}")

    if args.split_by == "unconditional":
        if args.horizon_s != _DEFAULT_HORIZON_S:
            print("  Note: --horizon-s is ignored with --split-by unconditional "
                  "(use --horizons instead).", file=sys.stderr)
        try:
            horizons = _parse_horizons(args.horizons)
        except argparse.ArgumentTypeError as exc:
            ap.error(str(exc))
        if not horizons:
            ap.error("--horizons produced no values")

        docs = _load(args.metrics_files)
        stats = _unconditional_stats(docs, horizons)

        print(f"Loaded {len(docs)} file(s): {', '.join(p.name for p, _ in docs)}")
        print(f"Horizons: {', '.join(_hfmt(h) for h in horizons)}")
        print("Split by: unconditional")
        print()
        _print_unconditional_table(horizons, stats)
        return

    if args.threshold is not None and args.split_by != "deviation":
        print("  Note: --threshold is ignored with --split-by role.", file=sys.stderr)

    docs = _load(args.metrics_files)
    rows_by_method, burden_map, seen_horizons = _rows_at_horizon(docs, args.horizon_s)

    if args.split_by == "deviation":
        result = _split_by_deviation(rows_by_method, args, docs)
    else:
        result = _split_by_role(rows_by_method, burden_map, args)

    if result is None:
        if seen_horizons:
            print(f"Horizons present in the loaded file(s): {seen_horizons}")
        else:
            print("No per_sample rows at all in the loaded file(s).")
        sys.exit(1)
    groups, header_lines = result

    print(f"Loaded {len(docs)} file(s): {', '.join(p.name for p, _ in docs)}")
    print(f"Horizon: {args.horizon_s} s")
    print(f"Split by: {args.split_by}")
    for line in header_lines:
        print(line)
    print()

    _print_table(groups, rows_by_method)


if __name__ == "__main__":
    main()
