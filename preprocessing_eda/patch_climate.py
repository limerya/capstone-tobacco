"""
Repair the climate columns in an existing aligned feature matrix.

align_stack.py treated a climate cell as valid when precip_total > 0, but
TerraClimate's land mask is slightly smaller than CHIRPS's, so some cells
kept zero-fill temperature / vpd / soil moisture. Bilinear resampling then
smeared those zeros up to ~4 km inland.

This rewrites only the climate columns of each Parquet part, using a
stricter validity rule. Everything else in the parts is untouched.

Run:
    python patch_climate.py "D:\\Projects\\capstone" "D:\\Projects\\capstone\\aligned"
"""

import os

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import re
import sys
import glob
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio

_db = glob.glob(os.path.join(os.path.dirname(rasterio.__file__), "**", "proj.db"),
                recursive=True)
if _db:
    os.environ["PROJ_DATA"] = os.path.dirname(_db[0])
    os.environ["PROJ_LIB"] = os.path.dirname(_db[0])

from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window

import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

TARGET_CRS = "EPSG:3121"
TARGET_RES = 30.0
WINDOW = 1024

CLIMATE_BANDS = ["precip_total", "precip_cv", "dry_spell_max", "tmean",
                 "tmin", "tmax", "vpd", "soil_moist"]
NEAREST = {"dry_spell_max"}
SCALE = {"tmean": 1e2, "tmin": 1e2, "tmax": 1e2, "vpd": 1e2,
         "precip_cv": 1e2, "soil_moist": 1e1}

PART = re.compile(r"^region(?P<region>\d+)_r(?P<row>\d+)_c(?P<col>\d+)\.parquet$")


def clean_climate(path, out_path):
    """Zero-fill -> NaN, using every band, not just precipitation."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        names = list(src.descriptions)
        profile = src.profile.copy()

    precip = arr[names.index("precip_total")]
    tmean = arr[names.index("tmean")]
    tmin = arr[names.index("tmin")]
    tmax = arr[names.index("tmax")]

    # a cell is valid only if BOTH sources have real data there
    valid = (precip > 0) & (tmean != 0) & (tmin != 0) & (tmax != 0)
    arr[:, ~valid] = np.nan

    profile.update(dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        dst.descriptions = tuple(names)
    return valid.sum(), valid.size


def build_template(terrain_path):
    with rasterio.open(terrain_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
            resolution=TARGET_RES)
    return {"transform": transform, "width": width, "height": height}


def warp_climate(paths, tpl, win):
    win_transform = rasterio.windows.transform(win, tpl["transform"])
    out = {}
    for b in CLIMATE_BANDS:
        dst = np.full((win.height, win.width), np.nan, dtype="float32")
        with rasterio.open(paths) as src:
            names = list(src.descriptions)
            if b not in names:
                continue
            reproject(
                source=rasterio.band(src, names.index(b) + 1),
                destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=win_transform, dst_crs=TARGET_CRS,
                dst_nodata=np.nan,
                resampling=Resampling.nearest if b in NEAREST else Resampling.bilinear,
            )
        out[b] = dst / SCALE[b] if b in SCALE else dst
    return out


def main():
    src_dir = r"D:\Projects\capstone"
    out_dir = r"D:\Projects\capstone\aligned"
    print("src_dir:", src_dir, os.path.isdir(src_dir))
    print("out_dir:", out_dir, os.path.isdir(out_dir))
    print("parts seen:", len(glob.glob(os.path.join(out_dir, "*.parquet"))))

    # locate source rasters
    terrain, climate = {}, defaultdict(dict)
    for p in glob.glob(os.path.join(src_dir, "**", "*.tif*"), recursive=True):
        if "_clean" in p or "aligned" in p.split(os.sep) or "_duplicates" in p:
            continue
        name = os.path.basename(p)
        m = re.match(r"^terrain_REGION_(\d)\.tif", name, re.I)
        if m:
            terrain[int(m.group(1))] = p
        m = re.match(r"^climate_REGION_(\d)_(\d{4}_\d{4})\.tif", name, re.I)
        if m:
            climate[int(m.group(1))][m.group(2)] = p

    tmp = os.path.join(out_dir, "_clean_fixed")
    os.makedirs(tmp, exist_ok=True)

    print("re-cleaning climate rasters with the stricter rule:")
    cleaned = defaultdict(dict)
    for region, seasons in climate.items():
        for season, path in sorted(seasons.items()):
            out = os.path.join(tmp, f"climate_R{region}_{season}.tif")
            keep, total = clean_climate(path, out)
            cleaned[region][season] = out
            print(f"  R{region} {season}: {keep}/{total} source cells valid")

    templates = {r: build_template(p) for r, p in terrain.items()}

    parts = sorted(glob.glob(os.path.join(out_dir, "*.parquet")))
    print(f"\npatching {len(parts)} parts")
    t0 = time.time()
    n_fixed = 0

    for i, part in enumerate(parts, 1):
        m = PART.match(os.path.basename(part))
        if not m:
            continue
        region = int(m.group("region"))
        row0, col0 = int(m.group("row")), int(m.group("col"))
        tpl = templates[region]

        h = min(WINDOW, tpl["height"] - row0)
        w = min(WINDOW, tpl["width"] - col0)
        win = Window(col0, row0, w, h)

        df = pd.read_parquet(part)
        rr = df["row"].to_numpy() - row0
        cc = df["col"].to_numpy() - col0

        stack = defaultdict(list)
        for season in sorted(cleaned[region]):
            d = warp_climate(cleaned[region][season], tpl, win)
            for b in CLIMATE_BANDS:
                if b in d:
                    stack[b].append(d[b])

        before_bad = int((df["tmean_med"] == 0).sum()) if "tmean_med" in df else 0

        for b in CLIMATE_BANDS:
            if not stack[b]:
                continue
            a = np.stack(stack[b])
            med = np.nanmedian(a, axis=0)
            iqr = np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0)
            df[f"{b}_med"] = med[rr, cc].astype("float32")
            df[f"{b}_iqr"] = iqr[rr, cc].astype("float32")

        after_nan = int(df["tmean_med"].isna().sum())
        n_fixed += before_bad
        df.to_parquet(part, compression="snappy", index=False)

        if i % 10 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)}  ({time.time() - t0:.0f}s)  "
                  f"last part: {before_bad} zero cells -> {after_nan} NaN")

    print(f"\ndone. {n_fixed:,} zero-filled climate cells replaced.")
    print("re-run sanity_check.py; tmean_med should no longer show a 0.000 minimum.")


if __name__ == "__main__":
    main()