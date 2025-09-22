import argparse
import os
import sys
import math
import re
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc

# Local utils
from src.config import load_config, ensure_config_dict_keys, resolve_config_path, DEFAULT_CONFIG_PATH, ConfigError

# -----------------------------
# Parsing helpers
# -----------------------------

def parse_voltage(val: Any) -> float:
    """Voltage is guaranteed to be numeric or numeric string; returns float or NaN."""
    if val is None:
        return float("nan")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    try:
        return float(str(val).strip())
    except Exception:
        return float("nan")
    

MULTILINESTRING_REGEX = re.compile(r'^\s*MULTILINESTRING\s*\(\s*(.+?)\s*\)\s*$', re.I | re.S)
PAIR_REGEX = re.compile(r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')  # (lon lat)

def parse_multilinestring(s: str) -> list[list[tuple[float, float]]]:
    """
    Parse MULTILINESTRING into list of segments.
    Returns [ [(lon,lat), ...], ... ], skipping segments with <2 points.
    """
    if not isinstance(s, str):
        return []
    m = MULTILINESTRING_REGEX.match(s.strip())
    if not m:
        return []
    inner = m.group(1)
    seg_strs = re.findall(r'\(\s*([^\)]*?)\s*\)', inner)
    lines: list[list[tuple[float, float]]] = []
    for seg in seg_strs:
        pts = [(float(x), float(y)) for (x, y) in PAIR_REGEX.findall(seg)]
        if len(pts) >= 2:
            lines.append(pts)
    return lines

# Parse geomtry column of Dataframe once, cache, filter, and collect failures
def prepare_geometries(df: pd.DataFrame, cfg: Dict[str, Any], *, debug: bool = False, n_preview: int = 20) -> tuple[pd.DataFrame, list[Any]]:
    """
    Parse MULTILINESTRING geometries ONCE, cache in df['geom_lines'], and keep only rows with ≥ 2 points total.
    Returns (df_valid, failed_parse_ids).

    - failed_parse_ids: rows where MULTILINESTRING parsing produced ZERO segments (true parse failure).
    - Rows with < 2 total points are excluded from visualization but not counted as parse failures.
    """
    fields = cfg["fields"]
    id_col = fields["id"]
    gcol = fields["geometry"]

    parsed_lines = df[gcol].map(parse_multilinestring)

    failed_mask = parsed_lines.map(lambda lines: len(lines) == 0)
    point_counts = parsed_lines.map(lambda lines: sum(len(seg) for seg in lines))
    valid_mask = point_counts >= 2

    failed_parse_ids = df.loc[failed_mask, id_col].tolist()

    df = df.copy()
    df["geom_lines"] = parsed_lines
    df["geom_point_count"] = point_counts

    if debug:
        total = len(df)
        n_failed = int(failed_mask.sum())
        n_valid = int(valid_mask.sum())
        n_short = total - n_failed - n_valid
        print("\n--------------------------------------")
        print(f"[GEOMETRY PARSE LOG] total={total} | parsed_ok={n_valid} | parse_failed={n_failed} | too_short(<2pts)={n_short}")
        print("--------------------------------------\n")
        if failed_parse_ids:
            print(f"[GEOM] parse_failed IDs (first {n_preview} of {len(failed_parse_ids)}): {failed_parse_ids[:n_preview]}")

    if valid_mask.sum() == 0:
        raise ValueError("No valid geometries (≥ 2 points) parsed from 'geometry' column.")

    return df.loc[valid_mask].reset_index(drop=True), failed_parse_ids

# -----------------------------
# CSV + config sanity checks
# -----------------------------

def sanity_check_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV is empty")
    return df

def sanity_check_fields(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    fields = cfg["fields"]
    missing = [col for col in fields.values() if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")

    # normalize voltage
    vcol = fields["voltage"]
    df[vcol] = df[vcol].apply(parse_voltage)

# -----------------------------
# Voltage bands + map helpers
# -----------------------------

def pick_voltage_band(bands: List[Dict[str, Any]], value: float) -> Dict[str, Any] | None:
    """Pick the first band where min ≤ value < max.
    If value is ≥ last band's min, return last band."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for b in bands:
        bmin, bmax = float(b["min"]), float(b["max"])
        if value >= bmin and value < bmax:
            return b
    last = bands[-1] if bands else None
    if last and value >= float(last.get("min", float("inf"))):
        return last
    return None

def compute_bounds(all_lines: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    """Compute bounding box (lon_min, lat_min, lon_max, lat_max) from all transmission lines.
    serves to center and zoom the map automatically on initial page load."""
    lons = [lon for line in all_lines for lon, _ in line]
    lats = [lat for line in all_lines for _, lat in line]
    return (min(lons), min(lats), max(lons), max(lats))


EARTH_RADIUS_KM = 6371
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Spherical surface distance between two coordinates in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0)**2
    return EARTH_RADIUS_KM * 2.0 * math.asin(math.sqrt(a))

def center_and_zoom(bbox: Tuple[float, float, float, float], cfg: Dict[str, Any]) -> Tuple[Dict[str, float], float]:
    """Given bounding box, compute center and zoom level.
    Adds coordinate padding around bbox if specified in config."""
    (lon_min, lat_min, lon_max, lat_max) = bbox
    pad = float(cfg["map"]["initial_view"].get("bbox_padding_deg", 0.1))
    lon_min -= pad; lon_max += pad; lat_min -= pad; lat_max += pad
    center = {"lon": (lon_min + lon_max) / 2.0, "lat": (lat_min + lat_max) / 2.0}

    # horizontal and vertical spans in KM using Haversine formula
    horiz_km = haversine_km(center["lat"], lon_min, center["lat"], lon_max)
    vert_km  = haversine_km(lat_min, center["lon"], lat_max, center["lon"])
    horiz_km = max(horiz_km, 1e-3)
    vert_km  = max(vert_km, 1e-3)

    span_km = max(horiz_km, vert_km)

    if   span_km > 5000: zoom = 3
    elif span_km > 2500: zoom = 4
    elif span_km > 1200: zoom = 5
    elif span_km > 600:  zoom = 6
    elif span_km > 300:  zoom = 7
    elif span_km > 150:  zoom = 8
    elif span_km > 75:   zoom = 9
    elif span_km > 35:   zoom = 10
    else: zoom = 11

    return center, zoom

# Print-once helper for debug reloader
def should_print_once(debug: bool) -> bool:
    """Dash/Flask reloader runs the script twice in debug. Print once in the child.
    """
    if not debug:
        return True
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"

# -----------------------------
# Figure builder
# -----------------------------

def build_figure(df: pd.DataFrame, cfg: Dict[str, Any]) -> go.Figure:
    fields = cfg["fields"]
    bands = cfg["voltage_bands"]
    vcol = fields["voltage"]

    all_lines: list[list[tuple[float, float]]] = []
    traces: list[go.Scattermap] = []
    shown_band_label: set[str] = set()
    line_width = int(cfg["map"]["line"].get("width", 2))

    use_cache = "geom_lines" in df.columns

    for _, row in df.iterrows():
        lines = row["geom_lines"] if use_cache else parse_multilinestring(row[fields["geometry"]])
        if not lines or sum(len(seg) for seg in lines) < 2:
            continue

        voltage = parse_voltage(row[vcol])
        band = pick_voltage_band(bands, voltage)
        color = (band or {}).get("color", "#888888")
        label = (band or {}).get("label", "Unknown")

        from_name = str(row.get(fields["from"], ""))
        to_name   = str(row.get(fields["to"],   ""))
        typ       = str(row.get(fields["type"],  ""))
        status    = str(row.get(fields["status"],""))

        hover = (
            f"<b>{from_name}</b> → <b>{to_name}</b><br>"
            f"Voltage: {voltage}<br>"
            f"Type: {typ}<br>"
            f"Status: {status}<br>"
            f"Band: {label}"
        )

        hover_template = hover + "<extra></extra>"
        
        for line in lines:
            lon = [p[0] for p in line]
            lat = [p[1] for p in line]
            traces.append(
                go.Scattermap(
                    lon=lon, lat=lat, mode="lines",
                    line=dict(color=color, width=line_width),
                    hovertemplate=hover_template,
                    name=label, legendgroup=label,
                    showlegend=(label not in shown_band_label),
                )
            )
            shown_band_label.add(label)
            all_lines.append(line)

    if not traces:
        raise ValueError("No map traces could be built from the CSV geometry column.")

    iview = cfg["map"]["initial_view"]
    if iview.get("method", "auto") == "fixed":
        center = {
            "lat": float(iview.get("fixed_center", {}).get("lat", 0)),
            "lon": float(iview.get("fixed_center", {}).get("lon", 0)),
        }
        zoom = float(iview.get("fixed_zoom", iview.get("fallback_zoom", 5)))
        print(f"[MAP VIEW: uses default] Using fixed center={center} zoom={zoom}")
    else:
        bbox = compute_bounds(all_lines)
        center, zoom = center_and_zoom(bbox, cfg)

    style = cfg["map"].get("style", "open-street-map")
    fig = go.Figure(data=traces)
    fig.update_layout(
        map=dict(
            style=style,
            center=dict(lat=center["lat"], lon=center["lon"]),
            zoom=float(zoom)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    
    debug_cfg = bool(cfg.get("app", {}).get("debug", True))
    print_once = should_print_once(debug_cfg)
    if debug_cfg and print_once:
        print("\n--------------------------------------")
        print(f"[MAP VIEW] center={center}  zoom={zoom}")
        print("--------------------------------------\n")
        
    return fig

# -----------------------------
# CLI args + main
# -----------------------------

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ROPES – Transmission Map (Dash)")
    p.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Config filename (looked up under ./config) OR any full/relative path. "
             "If omitted, behaves like 'default.json'.",
    )
    p.add_argument(
        "-c", "--config",
        dest="cfg_flag",
        default=None,
        help="Explicit config file (filename in ./config or any path).",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully merged config and exit.",
    )
    return p

def resolve_args_precedence(args) -> str | None:
    # Precedence: --config flag > positional > env > None (or) "default.json"
    chosen = args.cfg_flag or args.config or os.environ.get("ROPES_CONFIG")
    return chosen or "default.json"


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()

    override_hint = resolve_args_precedence(args)

    try:
        cfg, used_path = load_config(override_hint)
        cfg = ensure_config_dict_keys(cfg)
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    if args.print_config:
        import json as _json
        print(_json.dumps(cfg, indent=2))
        sys.exit(0)

    # Resolve CSV path
    csv_path = cfg.get("data", {}).get("csv_path", "data/sample.csv")
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(os.getcwd(), csv_path))

    # Load & validate CSV + fields
    df = sanity_check_csv(csv_path)
    sanity_check_fields(df, cfg)

    debug_cfg = bool(cfg.get("app", {}).get("debug", True))
    print_once = should_print_once(debug_cfg)

    # Parse geometries ONCE and keep only valid rows
    df_valid, failed_geom_ids = prepare_geometries(df, cfg, debug=(print_once and debug_cfg), n_preview=10)

    # Build figure
    fig = build_figure(df_valid, cfg)

    # Dash app
    app = Dash(__name__, assets_folder=os.path.join("src", "assets"))
    app.title = cfg.get("app", {}).get("title", "ROPES – Map")

    app.layout = html.Div(
        className="app-container",
        children=[
            html.H3(app.title, className="app-title"),
            html.Div("Legend shows one entry per voltage band; colors come from config.", className="legend-note"),
            dcc.Graph(id="map-graph", className="map-graph", figure=fig, config={"displayModeBar": True}),
        ],
    )


    host = cfg.get("app", {}).get("host", "0.0.0.0")
    port = int(cfg.get("app", {}).get("port", 8050))
    
    if debug_cfg and print_once:
            print("\n------ ROPES – Transmission Map ------")
            print(f"Using config: {used_path}")
            print(f"CSV: {csv_path}")
            print(f"Map style: {cfg['map'].get('style')}  |  Initial view: {cfg['map']['initial_view'].get('method', 'auto')}")
            print("Run modes:")
            print("  python app.py                       # uses config/default.json (via implicit override)")
            print("  python app.py my_config.json        # uses ./config/my_config.json if present")
            print("  python app.py -c /abs/path/config.json  # external path works too")
            print("--------------------------------------\n")

    app.run(host=host, port=port, debug=debug_cfg)

if __name__ == "__main__":
    main()
