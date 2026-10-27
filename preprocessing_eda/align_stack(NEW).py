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
  6. Applies unit conversions and writes Parquet parts.

Lean version -- changes from the original:
  - climate_rule requires precipitation AND tmean/tmin/tmax, the rule
    patch_climate.py applied afterwards. Climate comes out repaired, so
    patch_climate.py is NOT run after this script.
  - Only the bands finalize_matrix.py keeps are warped: 16 satellite bands,
    silt, tmin and tmax are no longer warped (the two cleaning rules still
    read silt/tmin/tmax from the raster itself, in clean_small).
  - slope is still warped, because slope_pct is derived from it; the
    degree column itself is not written.
  - IQR is computed only for the four IQR columns that are kept.
  - Median and IQR come from nan_quantiles(): one sort per band, results
    bit-identical to np.nanmedian / np.nanpercentile. np.nanpercentile
    loops pixel by pixel in Python whenever a NaN is present.
  - DROP_BANDS deleted (it was never read).
  - Output goes to aligned_lean/, so the existing aligned/ stays untouched
    until verify_lean_align.py passes.

Output: 35 columns per row (was 87); same cells, same rows.

Run:
    python align_stack.py "D:\\Projects\\capstone"
Then:
    python verify_lean_align.py "D:\\Projects\\capstone" aligned aligned_lean
"""

#IMMPORTANT remember: do not run patch_climate.py afterwards. The climate repair is now inside this script


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
OUT_DIR      = "aligned_lean"
AOI_FILE     = "aoi_regions.geojson"
ONLY         = set()         # e.g. {"region1_r0_c0.parquet"} to build just those windows

# Seasons. Climate has no 2025_2026 export, so it uses the first four.
SEASONS         = ["2021_2022", "2022_2023", "2023_2024", "2024_2025", "2025_2026"]
CLIMATE_SEASONS = ["2021_2022", "2022_2023", "2023_2024", "2024_2025"]

# The only IQR columns finalize_matrix.py keeps
# (solar_rad_iqr comes from patch_extras.py).
IQR_BANDS = {"precip_total", "tmean", "ndvi_med", "ndmi_med"}

# Scale divisors: stored_value / divisor = real value.
SCALE = {
    # index bands, stored x10000
    "ndvi_med": 1e4, "amplitude": 1e4,
    "water_freq": 1e4,
    "ndmi_med": 1e4, "ndmi_range": 1e4, "stress_freq": 1e4,
    # terrain
    "slope": 1e2,            # degrees x100
    "northness": 1e4, "eastness": 1e4,
    # soil
    "ph": 1e1,
    # climate
    "tmean": 1e2, "vpd": 1e2,
    "precip_cv": 1e2, "soil_moist": 1e1,
}
# soc is stored in OpenLandMap units of 5 g/kg -> multiply
MULTIPLY = {"soc": 5.0}

UNSCALED = {"elevation", "tpi", "sand", "clay", "awc",
            "precip_total", "dry_spell_max", "dss_peak", "n_valid"}
# Categorical / count bands use nearest-neighbour, everything else bilinear.
NEAREST = {"n_valid", "dss_peak", "dry_spell_max"}

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
        parts = path.split(os.sep)
        if "_duplicates" in path or OUT_DIR in parts or "aligned" in parts:
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
    # reads silt from the raster, so silt does not need to be warped
    tot = a[n.index("sand")] + a[n.index("clay")] + a[n.index("silt")]
    return tot > 0


def climate_rule(a, n):
    # CHIRPS (rain) and TerraClimate (temperature, vpd, soil moisture) have
    # different coastlines. A cell is valid only where BOTH have data -- the
    # same rule patch_climate.py applied afterwards. tmin/tmax are read from
    # the raster here, so they do not need to be warped.
    return ((a[n.index("precip_total")] > 0)
            & (a[n.index("tmean")] != 0)
            & (a[n.index("tmin")] != 0)
            & (a[n.index("tmax")] != 0))


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


def nan_quantiles(a, want_iqr=True):
    """
    Median, and optionally 25th/75th percentiles, along axis 0, ignoring NaN.
    One sort serves all three. Bit-identical to np.nanmedian(a, axis=0) and
    np.nanpercentile(a, [25, 75], axis=0) (default 'linear' method), but
    vectorised -- np.nanpercentile loops pixel by pixel whenever a NaN is
    present.
    """
    a = np.asarray(a)
    s = np.sort(a, axis=0)                          # NaN sorts to the end
    n = np.sum(~np.isnan(a), axis=0)                # valid seasons per pixel
    top = np.maximum(n - 1, 0)
    empty = n == 0

    def take(idx):
        return np.take_along_axis(s, np.clip(idx, 0, top)[None], axis=0)[0]

    # median: nanmedian averages the two middle values (same value twice if odd)
    h = n // 2
    lo = np.where(n % 2 == 1, h, h - 1)
    med = ((take(lo) + take(h)) / a.dtype.type(2)).astype(a.dtype)
    med[empty] = np.nan
    if not want_iqr:
        return med, None

    q = []
    for p in (25, 75):
        qf = np.true_divide(p, a.dtype.type(100))  # as numpy computes it
        vi = n * qf - qf                            # numpy's linear virtual index
        prev = np.floor(vi)
        g = (vi - prev).astype(a.dtype)
        A = take(prev.astype(np.intp))
        B = take(prev.astype(np.intp) + 1)
        d = B - A
        r = np.where(g >= 0.5, B - d * (1 - g), A + d * g).astype(a.dtype)
        r[empty] = np.nan
        q.append(r)
    return med, q[1] - q[0]


def summarise(cols, name, a, keep):
    """Write <name>_med, and <name>_iqr when that IQR is kept."""
    med, iqr = nan_quantiles(a, want_iqr=name in IQR_BANDS)
    cols[f"{name}_med"] = med[keep]
    if iqr is not None:
        cols[f"{name}_iqr"] = iqr[keep]


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

    # only what finalize_matrix.py keeps; n_valid is not written, but it
    # masks seasons with too few clear scenes
    sat_bands = {
        "ndvi": ["ndvi_med", "amplitude", "dss_peak", "n_valid"],
        "ndwi": ["water_freq", "n_valid"],
        "ndmi": ["ndmi_med", "ndmi_range", "stress_freq", "n_valid"],
    }
    terrain_bands = ["elevation", "slope", "northness", "eastness", "tpi"]
    soil_bands    = ["sand", "clay", "ph", "soc", "awc"]
    climate_bands = ["precip_total", "precip_cv", "dry_spell_max", "tmean",
                     "vpd", "soil_moist"]
    _warped = (set(terrain_bands) | set(soil_bands) | set(climate_bands)
               | {b for bs in sat_bands.values() for b in bs})
    _unknown = _warped - set(SCALE) - set(MULTIPLY) - UNSCALED
    assert not _unknown, f"band warped with no scaling decision: {sorted(_unknown)}"


    parts, n_rows, t0 = [], 0, time.time()
    n_win = ((tpl["height"] + WINDOW - 1) // WINDOW) * \
            ((tpl["width"] + WINDOW - 1) // WINDOW)
    k = 0

    for row in range(0, tpl["height"], WINDOW):
        for col in range(0, tpl["width"], WINDOW):
            k += 1
            part_name = f"region{region}_r{row}_c{col}.parquet"
            if ONLY and part_name not in ONLY:
                continue
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
            del cols["slope"]             # only slope_pct is used downstream

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
                summarise(cols, b, np.stack(stack[b]), keep)

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
                    summarise(cols, b, np.stack(stack[b]), keep)
                if product in ("ndvi", "ndmi"):      # finalize_matrix QUALITY
                    nv = np.stack(nvalid)
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

            part = os.path.join(out_dir, part_name)
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
    print("climate is already repaired: do NOT run patch_climate.py on this output.")


if __name__ == "__main__":
    main()