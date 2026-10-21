"""
Stage 1 - alignment and extraction.

Builds a co-registered analysis grid for Regions I and II and writes a
tabular feature matrix (Parquet) with one row per grid cell.

What it does
  1. Builds a 30 m template grid per region, in TARGET_CRS, from the
     terrain raster's footprint.
  2. Rasterises aoi_regions.geojson onto that grid -> region + province
     codes. Optionally erodes the mask by one cell to drop coastal
     cells that are part ocean.
  3. Pre-cleans the coarse layers (soil 250 m, climate ~4 km) by turning
     the zero fill into NaN BEFORE any resampling, so bilinear cannot
     smear zeros inland.
  4. Warps every layer onto the template, window by window.
  5. Collapses the five seasons (four for climate) into per-cell summary
     features: multi-season median and inter-season IQR.
  6. Applies unit conversions, drops redundant bands, writes Parquet parts.

Run:
    python align_stack.py "D:\\Projects\\capstone"
"""

import os
import re
import sys
import glob
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_geom
from rasterio.windows import Window
from rasterio.coords import disjoint_bounds

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
TARGET_CRS   = "EPSG:3121"   # PRS92 / Philippines zone III, as the chapter states
TARGET_RES   = 30.0          # metres
WINDOW       = 1024          # target-grid tile for the streaming loop
ERODE_MASK   = True          # drop one cell at the AOI edge (part-ocean cells)
MIN_N_VALID  = 3             # per season; fewer clear observations -> season dropped
SUBSAMPLE    = 1             # 1 = every cell; 3 = every 3rd row/col (smaller table)
OUT_DIR      = "aligned"
AOI_FILE     = "aoi_regions.geojson"

# Seasons. Climate has no 2025_2026 export, so it uses the first four.
SEASONS         = ["2021_2022", "2022_2023", "2023_2024", "2024_2025", "2025_2026"]
CLIMATE_SEASONS = ["2021_2022", "2022_2023", "2023_2024", "2024_2025"]

# Bands to drop outright.
#   drainage_proxy = (sand - clay) * 100, exactly. Redundant.
#   sos_dss / eos_dss are degenerate (mostly 0 / pinned to 168).
DROP_BANDS = {"drainage_proxy", "sos_dss", "eos_dss"}

# Scale divisors: stored_value / divisor = real value.
SCALE = {
    # index bands and derived spreads, stored x10000
    "ndvi_med": 1e4, "ndvi_p90": 1e4, "ndvi_p10": 1e4, "peak_ndvi": 1e4,
    "min_ndvi": 1e4, "amplitude": 1e4, "integral": 1e4,
    "ndwi_med": 1e4, "ndwi_min": 1e4, "ndwi_max": 1e4, "ndwi_p90": 1e4,
    "ndwi_range": 1e4, "water_freq": 1e4,
    "ndmi_med": 1e4, "ndmi_min": 1e4, "ndmi_max": 1e4, "ndmi_p10": 1e4,
    "ndmi_range": 1e4, "stress_freq": 1e4,
    # terrain
    "slope": 1e2,            # degrees x100
    "northness": 1e4, "eastness": 1e4,
    # soil
    "ph": 1e1,
    # climate
    "tmean": 1e2, "tmin": 1e2, "tmax": 1e2, "vpd": 1e2,
    "precip_cv": 1e2, "soil_moist": 1e1,
}
# soc is stored in OpenLandMap units of 5 g/kg -> multiply
MULTIPLY = {"soc": 5.0}

# Categorical / count bands use nearest-neighbour, everything else bilinear.
NEAREST = {"n_valid", "dss_max", "dss_min", "dss_peak", "dss_wettest", "dry_spell_max"}

FNAME = re.compile(
    r"^(?P<product>[a-z]+)_REGION_(?P<region>\d+)"
    r"(?:_(?P<s1>\d{4})_(?P<s2>\d{4}))?(?:_fast)?"
    r"(?:-(?P<yoff>\d{10})-(?P<xoff>\d{10}))?(?:-(?P<dup>\d+))?\.tiff?$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------
def discover(folder):
    """product -> region -> season -> [file paths]"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen_origin = set()
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.tif*"), recursive=True)):
        if "_duplicates" in path or OUT_DIR in path.split(os.sep):
            continue
        m = FNAME.match(os.path.basename(path))
        if not m:
            continue
        g = m.groupdict()
        product = g["product"].lower()
        region = int(g["region"])
        season = f"{g['s1']}_{g['s2']}" if g["s1"] else "static"
        origin = (product, region, season,
                  int(g["yoff"] or 0), int(g["xoff"] or 0))
        if origin in seen_origin:          # identical duplicate tile
            print(f"  skipping duplicate tile: {os.path.basename(path)}")
            continue
        seen_origin.add(origin)
        out[product][region][season].append(path)
    return out


# ----------------------------------------------------------------------
# template grid
# ----------------------------------------------------------------------
def build_template(terrain_path):
    with rasterio.open(terrain_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
            resolution=TARGET_RES)
        return {"crs": TARGET_CRS, "transform": transform,
                "width": width, "height": height, "src_crs": src.crs}


def load_aoi(folder):
    path = glob.glob(os.path.join(folder, "**", AOI_FILE), recursive=True)
    if not path:
        sys.exit(f"{AOI_FILE} not found under {folder}")
    with open(path[0], "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    return gj["features"]


def rasterise_aoi(features_list, tpl, region_code):
    """region mask + province index raster on the template grid."""
    shapes_region, shapes_prov, provinces = [], [], []
    for f in features_list:
        props = f.get("properties", {})
        if int(props.get("region", 0)) != region_code:
            continue
        geom = transform_geom("EPSG:4326", tpl["crs"], f["geometry"])
        name = props.get("province", "unknown")
        if name not in provinces:
            provinces.append(name)
        shapes_region.append((geom, 1))
        shapes_prov.append((geom, provinces.index(name) + 1))

    if not shapes_region:
        sys.exit(f"No AOI features with region == {region_code}. "
                 f"Check the 'region' property in {AOI_FILE}.")

    shape = (tpl["height"], tpl["width"])
    mask = features.rasterize(shapes_region, out_shape=shape,
                              transform=tpl["transform"], fill=0, dtype="uint8")
    prov = features.rasterize(shapes_prov, out_shape=shape,
                              transform=tpl["transform"], fill=0, dtype="uint8")

    if ERODE_MASK:
        m = mask.astype(bool)
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        mask = e.astype("uint8")
        prov = np.where(mask == 1, prov, 0).astype("uint8")

    return mask, prov, provinces


# ----------------------------------------------------------------------
# pre-clean the coarse layers (small enough to hold in memory)
# ----------------------------------------------------------------------
def clean_small(path, out_path, rule):
    """Write a float32 copy with the zero fill replaced by NaN."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        names = list(src.descriptions)
        profile = src.profile.copy()
    valid = rule(arr, names)
    arr[:, ~valid] = np.nan
    profile.update(dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        dst.descriptions = tuple(names)
    return out_path


def soil_rule(a, n):
    tot = a[n.index("sand")] + a[n.index("clay")] + a[n.index("silt")]
    return tot > 0


def climate_rule(a, n):
    return a[n.index("precip_total")] > 0


# ----------------------------------------------------------------------
# windowed warp
# ----------------------------------------------------------------------
def warp_window(paths, band_names, tpl, win, win_transform):
    """
    Read the given band names from a set of source tiles into the target
    window. Returns {band_name: 2-D float32 array with NaN for no data}.
    """
    h, w = win.height, win.width
    out = {b: np.full((h, w), np.nan, dtype="float32") for b in band_names}
    win_bounds = rasterio.windows.bounds(win, tpl["transform"])

    for path in paths:
        with rasterio.open(path) as src:
            src_bounds_tgt = rasterio.warp.transform_bounds(
                src.crs, tpl["crs"], *src.bounds)
            if disjoint_bounds(win_bounds, src_bounds_tgt):
                continue
            names = list(src.descriptions)
            for b in band_names:
                if b not in names:
                    continue
                tmp = np.full((h, w), np.nan, dtype="float32")
                rs = Resampling.nearest if b in NEAREST else Resampling.bilinear
                reproject(
                    source=rasterio.band(src, names.index(b) + 1),
                    destination=tmp,
                    src_transform=src.transform, src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=win_transform, dst_crs=tpl["crs"],
                    dst_nodata=np.nan, resampling=rs,
                )
                np.copyto(out[b], tmp, where=np.isnan(out[b]))
    return out


def rescale(name, arr):
    if name in SCALE:
        arr = arr / SCALE[name]
    if name in MULTIPLY:
        arr = arr * MULTIPLY[name]
    return arr


# ----------------------------------------------------------------------
# main per-region routine
# ----------------------------------------------------------------------
def process_region(folder, files, region, out_dir):
    print(f"\n{'=' * 70}\nREGION {region}\n{'=' * 70}")

    terrain_path = files["terrain"][region]["static"][0]
    tpl = build_template(terrain_path)
    print(f"template: {tpl['width']} x {tpl['height']} @ {TARGET_RES} m "
          f"in {TARGET_CRS}")

    aoi = load_aoi(folder)
    mask, prov, prov_names = rasterise_aoi(aoi, tpl, region)
    print(f"AOI cells: {int(mask.sum()):,}  provinces: {prov_names}")

    tmp_dir = os.path.join(out_dir, "_clean")
    os.makedirs(tmp_dir, exist_ok=True)
    soil_clean = clean_small(
        files["soil"][region]["static"][0],
        os.path.join(tmp_dir, f"soil_R{region}.tif"), soil_rule)
    climate_clean = {
        s: clean_small(files["climate"][region][s][0],
                       os.path.join(tmp_dir, f"climate_R{region}_{s}.tif"),
                       climate_rule)
        for s in CLIMATE_SEASONS if s in files["climate"][region]
    }

    sat_bands = {
        "ndvi": ["ndvi_med", "ndvi_p90", "ndvi_p10", "peak_ndvi", "min_ndvi",
                 "amplitude", "dss_peak", "integral", "n_valid"],
        "ndwi": ["ndwi_med", "ndwi_min", "ndwi_max", "ndwi_p90", "ndwi_range",
                 "water_freq", "dss_wettest", "n_valid"],
        "ndmi": ["ndmi_med", "ndmi_min", "ndmi_max", "ndmi_p10", "ndmi_range",
                 "stress_freq", "dss_max", "dss_min", "n_valid"],
    }
    terrain_bands = ["elevation", "slope", "northness", "eastness", "tpi"]
    soil_bands    = ["sand", "clay", "silt", "ph", "soc", "awc"]
    climate_bands = ["precip_total", "precip_cv", "dry_spell_max", "tmean",
                     "tmin", "tmax", "vpd", "soil_moist"]

    parts, n_rows, t0 = [], 0, time.time()
    n_win = ((tpl["height"] + WINDOW - 1) // WINDOW) * \
            ((tpl["width"] + WINDOW - 1) // WINDOW)
    k = 0

    for row in range(0, tpl["height"], WINDOW):
        for col in range(0, tpl["width"], WINDOW):
            k += 1
            h = min(WINDOW, tpl["height"] - row)
            w = min(WINDOW, tpl["width"] - col)
            win = Window(col, row, w, h)
            sub_mask = mask[row:row + h, col:col + w]
            if not sub_mask.any():
                continue
            win_transform = rasterio.windows.transform(win, tpl["transform"])
            keep = sub_mask.astype(bool)
            if SUBSAMPLE > 1:
                stride = np.zeros_like(keep)
                stride[::SUBSAMPLE, ::SUBSAMPLE] = True
                keep &= stride

            cols = {}

            # --- terrain (static) -------------------------------------
            t = warp_window(files["terrain"][region]["static"], terrain_bands,
                            tpl, win, win_transform)
            for b in terrain_bands:
                cols[b] = rescale(b, t[b])[keep]
            # slope: degrees -> percent, as Table 3.1 specifies
            cols["slope_pct"] = np.tan(np.radians(cols["slope"])) * 100.0

            # --- soil (static) ----------------------------------------
            s = warp_window([soil_clean], soil_bands, tpl, win, win_transform)
            for b in soil_bands:
                cols[b] = rescale(b, s[b])[keep]

            # --- climate (per season -> median / IQR) -----------------
            stack = defaultdict(list)
            for season, path in climate_clean.items():
                c = warp_window([path], climate_bands, tpl, win, win_transform)
                for b in climate_bands:
                    stack[b].append(rescale(b, c[b]))
            for b in climate_bands:
                a = np.stack(stack[b])
                cols[f"{b}_med"] = np.nanmedian(a, axis=0)[keep]
                cols[f"{b}_iqr"] = (np.nanpercentile(a, 75, axis=0) -
                                    np.nanpercentile(a, 25, axis=0))[keep]

            # --- satellite (per season -> median / IQR) ---------------
            for product, bands in sat_bands.items():
                stack = defaultdict(list)
                nvalid = []
                for season in SEASONS:
                    paths = files[product][region].get(season)
                    if not paths:
                        continue
                    d = warp_window(paths, bands, tpl, win, win_transform)
                    nv = d["n_valid"]
                    nvalid.append(nv)
                    poor = nv < MIN_N_VALID          # too few clear scenes
                    for b in bands:
                        if b == "n_valid":
                            continue
                        arr = rescale(b, d[b]).copy()
                        arr[poor] = np.nan
                        stack[b].append(arr)
                for b in bands:
                    if b == "n_valid":
                        continue
                    a = np.stack(stack[b])
                    cols[f"{b}_med"] = np.nanmedian(a, axis=0)[keep]
                    cols[f"{b}_iqr"] = (np.nanpercentile(a, 75, axis=0) -
                                        np.nanpercentile(a, 25, axis=0))[keep]
                nv = np.stack(nvalid)
                cols[f"{product}_n_valid_min"] = np.nanmin(nv, axis=0)[keep]
                cols[f"{product}_seasons_ok"] = np.nansum(
                    nv >= MIN_N_VALID, axis=0)[keep]

            # --- coordinates and ids ----------------------------------
            rr, cc = np.nonzero(keep)
            xs, ys = rasterio.transform.xy(win_transform, rr, cc)
            cols["x"] = np.asarray(xs, dtype="float64")
            cols["y"] = np.asarray(ys, dtype="float64")
            cols["row"] = (row + rr).astype("int32")
            cols["col"] = (col + cc).astype("int32")
            cols["region"] = np.full(rr.size, region, dtype="int8")
            pv = prov[row:row + h, col:col + w][keep]
            cols["province"] = pd.Categorical.from_codes(
                np.clip(pv.astype(int) - 1, 0, len(prov_names) - 1),
                categories=prov_names)

            df = pd.DataFrame(cols)
            for c in df.columns:
                if df[c].dtype == "float64" and c not in ("x", "y"):
                    df[c] = df[c].astype("float32")

            part = os.path.join(out_dir, f"region{region}_r{row}_c{col}.parquet")
            df.to_parquet(part, compression="snappy", index=False)
            parts.append(part)
            n_rows += len(df)

            if k % 10 == 0 or k == n_win:
                el = time.time() - t0
                print(f"  window {k}/{n_win}  rows so far {n_rows:,}  "
                      f"{el / 60:.1f} min")

    print(f"region {region}: {n_rows:,} rows in {len(parts)} parts")
    return parts, n_rows


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\Projects\capstone"
    out_dir = os.path.join(folder, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    files = discover(folder)
    for product in ("terrain", "soil", "climate", "ndvi", "ndwi", "ndmi"):
        if product not in files:
            sys.exit(f"missing product: {product}")

    total = 0
    for region in sorted(files["terrain"].keys()):
        _, n = process_region(folder, files, region, out_dir)
        total += n

    print(f"\ndone: {total:,} rows -> {out_dir}")
    print("load with: pd.read_parquet(r'%s')" % out_dir)


if __name__ == "__main__":
    main()