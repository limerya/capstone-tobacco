"""
Audit GEE exports in a local folder.

Reports, per (product, region, season):
  - number of files, duplicate tiles, unique tiles
  - full pixel grid implied by the tile offsets
  - percentage of that grid present on disk, and missing tile origins
  - union bounds in degrees
Then checks extent consistency per region, and samples band values
so you can verify scale factors / units.

Usage:
    python audit_exports.py "D:\\Projects\\capstone"
"""

import os
import re
import sys
import glob
from collections import defaultdict

import numpy as np
import rasterio

FNAME = re.compile(
    r"^(?P<product>[a-z]+)_REGION_(?P<region>\d+)"
    r"(?:_(?P<s1>\d{4})_(?P<s2>\d{4}))?"
    r"(?:_fast)?"
    r"(?:-(?P<yoff>\d{10})-(?P<xoff>\d{10}))?"
    r"(?:-(?P<dup>\d+))?"
    r"\.tiff?$",
    re.IGNORECASE,
)


def scan(folder):
    files = glob.glob(os.path.join(folder, "**", "*.tif*"), recursive=True)
    groups = defaultdict(list)
    unparsed = []

    for path in sorted(files):
        if "_duplicates" in path:
            continue
        m = FNAME.match(os.path.basename(path))
        if not m:
            unparsed.append(path)
            continue
        g = m.groupdict()
        season = f"{g['s1']}_{g['s2']}" if g["s1"] else "static"
        key = (g["product"].lower(), f"R{g['region']}", season)

        with rasterio.open(path) as src:
            groups[key].append({
                "path": path,
                "name": os.path.basename(path),
                "yoff": int(g["yoff"]) if g["yoff"] else 0,
                "xoff": int(g["xoff"]) if g["xoff"] else 0,
                "w": src.width,
                "h": src.height,
                "res": src.res,
                "bounds": tuple(src.bounds),
                "bands": src.descriptions,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
            })

    return groups, unparsed


def report(groups):
    print("=" * 100)
    print("COVERAGE AUDIT")
    print("=" * 100)

    for key in sorted(groups):
        recs = groups[key]
        product, region, season = key

        by_origin = defaultdict(list)
        for r in recs:
            by_origin[(r["yoff"], r["xoff"])].append(r)
        dups = {k: v for k, v in by_origin.items() if len(v) > 1}

        grid_h = max(r["yoff"] + r["h"] for r in recs)
        grid_w = max(r["xoff"] + r["w"] for r in recs)

        ys = sorted({r["yoff"] for r in recs})
        xs = sorted({r["xoff"] for r in recs})
        tile_y = min((b - a for a, b in zip(ys, ys[1:])), default=grid_h)
        tile_x = min((b - a for a, b in zip(xs, xs[1:])), default=grid_w)

        expected = [(y, x) for y in range(0, grid_h, tile_y)
                    for x in range(0, grid_w, tile_x)]
        missing = [o for o in expected if o not in by_origin]

        covered = sum(v[0]["w"] * v[0]["h"] for v in by_origin.values())
        pct = 100.0 * covered / (grid_w * grid_h)

        res = recs[0]["res"][0]
        lons = [c for r in recs for c in (r["bounds"][0], r["bounds"][2])]
        lats = [c for r in recs for c in (r["bounds"][1], r["bounds"][3])]

        flag = "" if pct > 99.5 and not missing and not dups else "   <-- CHECK"
        print(f"\n{product:8s} {region:3s} {season:10s} files={len(recs):2d} "
              f"tiles={len(by_origin):2d} coverage={pct:6.1f}%{flag}")
        print(f"    grid {grid_w} x {grid_h} px @ ~{res * 111320:.0f} m | "
              f"lon {min(lons):.4f}..{max(lons):.4f} | "
              f"lat {min(lats):.4f}..{max(lats):.4f}")
        print(f"    dtype={recs[0]['dtype']} nodata={recs[0]['nodata']}")

        for origin, v in sorted(dups.items()):
            print(f"    DUPLICATE at {origin}: " + ", ".join(x["name"] for x in v))
        if missing:
            print(f"    MISSING tile origins ({len(missing)}): {missing[:12]}")


def check_extents(groups):
    print("\n" + "=" * 100)
    print("EXTENT CONSISTENCY PER REGION (lon_min, lat_min, lon_max, lat_max)")
    print("=" * 100)

    per_region = defaultdict(lambda: defaultdict(list))
    for (product, region, season), recs in groups.items():
        lons = [c for r in recs for c in (r["bounds"][0], r["bounds"][2])]
        lats = [c for r in recs for c in (r["bounds"][1], r["bounds"][3])]
        box = (round(min(lons), 2), round(min(lats), 2),
               round(max(lons), 2), round(max(lats), 2))
        per_region[region][box].append(f"{product}_{season}")

    for region in sorted(per_region):
        print(f"\n{region}:")
        for box, names in per_region[region].items():
            print(f"    {box}  <- {', '.join(sorted(names))}")
        if len(per_region[region]) > 1:
            print("    ^ extents differ (coarse layers differ slightly by design; "
                  "look for large gaps)")


def sample_bands(groups, target_px=200_000):
    print("\n" + "=" * 100)
    print("BAND VALUE RANGES (verify units / scale factors)")
    print("=" * 100)

    seen = set()
    for key in sorted(groups):
        product = key[0]
        if product in seen:
            continue
        seen.add(product)

        rec = max(groups[key], key=lambda r: r["w"] * r["h"])
        with rasterio.open(rec["path"]) as src:
            f = max(1, int(np.sqrt(src.width * src.height / target_px)))
            shape = (src.count, max(1, src.height // f), max(1, src.width // f))
            arr = src.read(out_shape=shape, masked=True).astype("float64")
            names = src.descriptions

        print(f"\n{product}  ({rec['name']}, 1/{f} decimation)")
        for i, band in enumerate(names):
            b = arr[i].compressed()
            b = b[np.isfinite(b)]
            if b.size == 0:
                print(f"    {str(band):14s} no valid values")
                continue
            zeros = 100 * np.mean(b == 0)
            print(f"    {str(band):14s} min={b.min():11.3f}  "
                  f"p50={np.median(b):11.3f}  max={b.max():11.3f}  "
                  f"valid={100 * b.size / arr[i].size:5.1f}%  zeros={zeros:5.1f}%")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\Projects\capstone"
    groups, unparsed = scan(folder)
    if not groups:
        print(f"No rasters matched under {folder}")
        sys.exit(1)

    report(groups)
    check_extents(groups)
    sample_bands(groups)

    if unparsed:
        print("\nUnmatched filenames:")
        for p in unparsed:
            print("   ", os.path.basename(p))