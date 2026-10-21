"""
Compute a proper topographic wetness index from the exported DEM.

TWI = ln(a / tan(beta)), where a is specific contributing area from
D-infinity flow routing and beta is local slope. This replaces the
inverted-TPI proxy, which correlated -0.95 with slope and so carried
almost no independent drainage information.

Pipeline:
  breach depressions -> D-inf flow accumulation (SCA) -> slope -> TWI
Output is written as twi_REGION_<n>.tif, stored x100 as int16, matching
what patch_extras.py expects.

Needs:  pip install whitebox rasterio numpy
Run:    python compute_twi.py
"""

import os
import glob
import re
import time

import numpy as np
import rasterio

SRC_DIR = r"D:\Projects\capstone"
WORK = os.path.join(SRC_DIR, "twi_work")
TWI_SCALE = 100.0          # stored x100 as int16


def find_dems():
    out = {}
    for p in glob.glob(os.path.join(SRC_DIR, "**", "dem_REGION_*.tif"),
                       recursive=True):
        if "twi_work" in p:
            continue
        m = re.search(r"dem_REGION_(\d)", os.path.basename(p))
        if m:
            out.setdefault(int(m.group(1)), []).append(p)
    return out


def mosaic_if_needed(paths, out_path):
    """GEE may split a large export into tiles; merge them first."""
    if len(paths) == 1:
        return paths[0]
    from rasterio.merge import merge
    srcs = [rasterio.open(p) for p in paths]
    arr, transform = merge(srcs)
    profile = srcs[0].profile.copy()
    profile.update(height=arr.shape[1], width=arr.shape[2],
                   transform=transform, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
    for s in srcs:
        s.close()
    return out_path


def prepare_dem(path, out_path):
    """Zero fill over sea -> nodata, so it is not treated as flat land."""
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        profile = src.profile.copy()
    a[a == 0] = -32768.0
    profile.update(dtype="float32", nodata=-32768.0, compress="deflate",
                   count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(a, 1)
    return out_path


def main():
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(WORK)

    os.makedirs(WORK, exist_ok=True)
    dems = find_dems()
    if not dems:
        raise SystemExit("no dem_REGION_*.tif found - run the GEE export first")

    for region, paths in sorted(dems.items()):
        print(f"\n=== Region {region} ===")
        t0 = time.time()

        merged = mosaic_if_needed(paths, os.path.join(WORK, f"dem_r{region}.tif"))
        dem = prepare_dem(merged, os.path.join(WORK, f"dem_clean_r{region}.tif"))
        with rasterio.open(dem) as s:
            print(f"DEM {s.width} x {s.height} @ {s.res[0]:.1f} m, {s.crs}")

        base = os.path.basename(dem)
        filled = f"filled_r{region}.tif"
        sca    = f"sca_r{region}.tif"
        slope  = f"slope_r{region}.tif"
        twi    = f"twi_raw_r{region}.tif"

        print("  breaching depressions...")
        wbt.breach_depressions_least_cost(base, filled, dist=100, fill=True)

        print("  D-infinity flow accumulation...")
        wbt.d_inf_flow_accumulation(filled, sca,
                                    out_type="Specific Contributing Area")

        print("  slope...")
        wbt.slope(filled, slope, units="degrees")

        print("  wetness index...")
        wbt.wetness_index(sca, slope, twi)

        # convert to int16 x100 with the naming patch_extras expects
        src_twi = os.path.join(WORK, twi)
        out_twi = os.path.join(SRC_DIR, f"twi_REGION_{region}.tif")
        with rasterio.open(src_twi) as s:
            a = s.read(1).astype("float32")
            profile = s.profile.copy()
        bad = ~np.isfinite(a)
        if s.nodata is not None:
            bad |= (a == s.nodata)
        a = np.clip(a, -20, 40)
        v = np.where(bad, 0, np.round(a * TWI_SCALE)).astype("int16")

        profile.update(dtype="int16", nodata=0, count=1, compress="deflate")
        with rasterio.open(out_twi, "w", **profile) as dst:
            dst.write(v, 1)
            dst.descriptions = ("twi",)

        good = v[v != 0] / TWI_SCALE
        print(f"  -> {out_twi}")
        print(f"     TWI p5={np.percentile(good,5):.2f} "
              f"p50={np.median(good):.2f} p95={np.percentile(good,95):.2f} "
              f"({(time.time()-t0)/60:.1f} min)")

    print("\nnow delete aligned_extras and re-run patch_extras.py")
    print(f"(intermediates kept in {WORK}; safe to delete afterwards)")


if __name__ == "__main__":
    main()