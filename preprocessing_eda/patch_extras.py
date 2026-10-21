"""
Add CEC, nitrogen, TWI and solar radiation to the aligned grid.  (v2)

Change from v1: ERA5-Land is a land-only reanalysis at ~11 km, so its
coarse land mask drops cells within about one grid step of the coast.
That removed 22.5% of solar values - concentrated in exactly the coastal
lowlands where Virginia tobacco is grown (Cagayan 45%, Isabela 22%,
Pangasinan 18%), while the two landlocked provinces lost nothing.

Solar values are therefore extended outward into the masked gap by
iterative nearest-neighbour fill before resampling. Solar radiation
varies smoothly at this scale (13-20 MJ/m2/day across the study area),
so extending land values a few grid cells seaward introduces little
error. This is the source-appropriate gap filling Section 3.4.1
describes. Only solar is filled; CEC, nitrogen and TWI are not.

Writes a SEPARATE small part per window holding only the new columns
plus the join keys, so the 12 GB of existing parts are never rewritten.

Run:
    python patch_extras_v2.py
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

# ----------------------------------------------------------------- CONFIG
SRC_DIR = r"D:\Projects\capstone"
OUT_DIR = r"D:\Projects\capstone\aligned"
EXTRA_DIR = r"D:\Projects\capstone\aligned_extras"

TARGET_CRS = "EPSG:3121"
TARGET_RES = 30.0
WINDOW = 1024

CEC_DIV = 10.0          # mmol(c)/kg -> cmol(c)/kg
NITROGEN_DIV = 1000.0   # verified against C:N ratio (15.5)
TWI_DIV = 100.0
SOLAR_DIV = 100.0       # -> MJ/m2/day

SOLAR_FILL_ITERS = 12   # ~12 cells at 11 km covers the coastal mask gap

SOLAR_SEASONS = ["2021_2022", "2022_2023", "2023_2024", "2024_2025", "2025_2026"]

PART = re.compile(r"^region(?P<region>\d+)_r(?P<row>\d+)_c(?P<col>\d+)\.parquet$")


def fill_nearest(arr, max_iter=SOLAR_FILL_ITERS):
    """Extend valid values outward into NaN gaps, one ring per iteration."""
    out = arr.copy()
    for _ in range(max_iter):
        nan = np.isnan(out)
        if not nan.any():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.full_like(out, np.nan)
            ys = slice(max(dy, 0), out.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), out.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), out.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), out.shape[1] + min(-dx, 0))
            sh[yd, xd] = out[ys, xs]
            ok = ~np.isnan(sh)
            acc[ok] += sh[ok]
            cnt[ok] += 1
        filled = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        out[nan] = filled[nan]
    return out


def clean_zeros(path, out_path, fill=False):
    """Zero fill -> NaN before resampling; optionally extend into the gap."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        names = list(src.descriptions)
        profile = src.profile.copy()

    valid = np.all(arr != 0, axis=0)
    arr[:, ~valid] = np.nan
    n_valid = int(valid.sum())

    if fill:
        for i in range(arr.shape[0]):
            arr[i] = fill_nearest(arr[i])
        n_after = int(np.isfinite(arr[0]).sum())
    else:
        n_after = n_valid

    profile.update(dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        dst.descriptions = tuple(names)
    return n_valid, n_after, int(valid.size)


def build_template(terrain_path):
    with rasterio.open(terrain_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
            resolution=TARGET_RES)
    return {"transform": transform, "width": width, "height": height}


def warp_band(path, band, tpl, win):
    win_transform = rasterio.windows.transform(win, tpl["transform"])
    dst = np.full((win.height, win.width), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        if band not in names:
            return dst
        reproject(
            source=rasterio.band(src, names.index(band) + 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
            dst_transform=win_transform, dst_crs=TARGET_CRS, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return dst


def find(pattern):
    return sorted(p for p in glob.glob(os.path.join(SRC_DIR, "**", pattern),
                                       recursive=True)
                  if "_cleanx" not in p
                  and "aligned" not in p.split(os.sep)
                  and "twi_work" not in p.split(os.sep))


def main():
    os.makedirs(EXTRA_DIR, exist_ok=True)
    tmp = os.path.join(EXTRA_DIR, "_cleanx")
    os.makedirs(tmp, exist_ok=True)

    terrain = {}
    for p in find("terrain_REGION_*.tif"):
        terrain[int(re.search(r"REGION_(\d)", p).group(1))] = p
    if not terrain:
        sys.exit("terrain rasters not found")

    print("cleaning zero fill before resampling:")
    soilx, twi, solar = {}, {}, defaultdict(dict)

    for p in find("soilextra_REGION_*.tif"):
        r = int(re.search(r"REGION_(\d)", p).group(1))
        out = os.path.join(tmp, f"soilextra_R{r}.tif")
        k, _, t = clean_zeros(p, out)
        soilx[r] = out
        print(f"  soilextra R{r}: {k:,}/{t:,} valid")

    for p in find("twi_REGION_*.tif"):
        r = int(re.search(r"REGION_(\d)", p).group(1))
        out = os.path.join(tmp, f"twi_R{r}.tif")
        k, _, t = clean_zeros(p, out)
        twi[r] = out
        print(f"  twi R{r}: {k:,}/{t:,} valid")

    for p in find("solar_REGION_*.tif"):
        m = re.search(r"REGION_(\d)_(\d{4}_\d{4})", p)
        r, s = int(m.group(1)), m.group(2)
        out = os.path.join(tmp, f"solar_R{r}_{s}.tif")
        k, a, t = clean_zeros(p, out, fill=True)
        solar[r][s] = out
        print(f"  solar R{r} {s}: {k:,} valid -> {a:,} after coastal fill "
              f"(of {t:,})")

    templates = {r: build_template(p) for r, p in terrain.items()}

    parts = sorted(glob.glob(os.path.join(OUT_DIR, "*.parquet")))
    print(f"\nwriting extras for {len(parts)} windows -> {EXTRA_DIR}")
    t0 = time.time()
    missing = defaultdict(int)
    total = 0

    for i, part in enumerate(parts, 1):
        m = PART.match(os.path.basename(part))
        if not m:
            continue
        region = int(m.group("region"))
        row0, col0 = int(m.group("row")), int(m.group("col"))
        out_part = os.path.join(EXTRA_DIR, os.path.basename(part))
        if os.path.exists(out_part):
            continue

        tpl = templates[region]
        h = min(WINDOW, tpl["height"] - row0)
        w = min(WINDOW, tpl["width"] - col0)
        win = Window(col0, row0, w, h)

        keys = pd.read_parquet(part, columns=["region", "row", "col"])
        rr = keys["row"].to_numpy() - row0
        cc = keys["col"].to_numpy() - col0

        out = {"region": keys["region"].to_numpy(),
               "row": keys["row"].to_numpy(),
               "col": keys["col"].to_numpy()}

        out["cec"] = (warp_band(soilx[region], "cec", tpl, win)
                      / CEC_DIV)[rr, cc].astype("float32")
        out["nitrogen"] = (warp_band(soilx[region], "nitrogen", tpl, win)
                           / NITROGEN_DIV)[rr, cc].astype("float32")
        if region in twi:
            out["twi"] = (warp_band(twi[region], "twi", tpl, win)
                          / TWI_DIV)[rr, cc].astype("float32")

        stack = [warp_band(solar[region][s], "solar_rad", tpl, win) / SOLAR_DIV
                 for s in SOLAR_SEASONS if s in solar[region]]
        if stack:
            a = np.stack(stack)
            out["solar_rad_med"] = np.nanmedian(a, axis=0)[rr, cc].astype("float32")
            out["solar_rad_iqr"] = (np.nanpercentile(a, 75, axis=0) -
                                    np.nanpercentile(a, 25, axis=0))[rr, cc].astype("float32")

        df = pd.DataFrame(out)
        for c in ("twi", "solar_rad_med", "cec"):
            if c in df:
                missing[c] += int(df[c].isna().sum())
        total += len(df)
        df.to_parquet(out_part, compression="snappy", index=False)

        if i % 10 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)}  {total:,} rows  "
                  f"({(time.time() - t0) / 60:.1f} min)")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min, {total:,} rows")
    for c, n in missing.items():
        print(f"  {c}: {n:,} missing ({100 * n / max(total, 1):.2f}%)")
    print(f"\nextras in {EXTRA_DIR}; merge on (region, row, col)")


if __name__ == "__main__":
    main()