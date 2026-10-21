"""
Compare SoilGrids texture against OpenLandMap on the same cells.

The question is narrow: does SoilGrids resolve texture variation that
OpenLandMap smooths away? Three things decide it --

  1. spread      -- a wider sand distribution means more signal
  2. top end     -- Table 3.1 wants light/sandy surface soils, so what
                    matters is whether ANY cells reach 50-60% sand
  3. geography   -- the high-sand cells should fall on the coastal
                    plains and river terraces, not scattered at random

Run after downloading sgtexture_REGION_1.tif and sgtexture_REGION_2.tif.

    python compare_texture.py
"""

import os

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import glob
import numpy as np
import pandas as pd
import rasterio

_db = glob.glob(os.path.join(os.path.dirname(rasterio.__file__), "**", "proj.db"),
                recursive=True)
if _db:
    os.environ["PROJ_DATA"] = os.path.dirname(_db[0])
    os.environ["PROJ_LIB"] = os.path.dirname(_db[0])

SRC = r"D:\Projects\capstone"
SAMPLE = r"D:\Projects\capstone\training_sample_labeled.parquet"

PCTS = [1, 5, 25, 50, 75, 95, 99]


def read_sg():
    out = {}
    for p in sorted(glob.glob(os.path.join(SRC, "**", "sgtexture_REGION_*.tif"),
                              recursive=True)):
        if "_clean" in p:
            continue
        r = int(p.split("REGION_")[1][0])
        with rasterio.open(p) as s:
            a = s.read().astype("float64")
            names = list(s.descriptions)
        # g/kg -> percent; zero fill -> NaN
        valid = np.all(a != 0, axis=0)
        d = {n: np.where(valid, a[i] / 10.0, np.nan)
             for i, n in enumerate(names)}
        out[r] = d
        print(f"  loaded {os.path.basename(p)}  {a.shape[1]}x{a.shape[2]}  "
              f"{int(valid.sum()):,} valid cells")
    return out


def describe(name, v):
    v = v[np.isfinite(v)]
    q = np.percentile(v, PCTS)
    print(f"  {name:22s} n={v.size:>10,}  mean={v.mean():5.1f}  sd={v.std():5.1f}  "
          + "  ".join(f"p{p}={x:.1f}" for p, x in zip(PCTS, q)))
    return v


def main():
    print("SoilGrids rasters:")
    sg = read_sg()
    if not sg:
        raise SystemExit("no sgtexture_REGION_*.tif found -- run the GEE export")

    print("\nOpenLandMap (from the labelled sample):")
    smp = pd.read_parquet(SAMPLE, columns=["sand", "clay", "region", "elevation"])

    print("\n" + "=" * 96)
    print("SAND CONTENT, % -- the factor that decides the texture rating")
    print("=" * 96)
    olm_sand = describe("OpenLandMap sand", smp["sand"].to_numpy("float64"))
    sg_sand = describe("SoilGrids sand",
                       np.concatenate([d["sg_sand"].ravel() for d in sg.values()]))

    print("\n" + "=" * 96)
    print("CLAY CONTENT, %")
    print("=" * 96)
    describe("OpenLandMap clay", smp["clay"].to_numpy("float64"))
    describe("SoilGrids clay",
             np.concatenate([d["sg_clay"].ravel() for d in sg.values()]))

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    for label, v in [("OpenLandMap", olm_sand), ("SoilGrids", sg_sand)]:
        v = v[np.isfinite(v)]
        print(f"  {label:12s} sd={v.std():5.2f}  "
              f">=50% sand: {100 * (v >= 50).mean():6.3f}%  "
              f">=60% sand: {100 * (v >= 60).mean():6.3f}%  "
              f"range p1-p99: {np.percentile(v, 1):.1f}-{np.percentile(v, 99):.1f}")
    print()
    print("  A useful texture layer needs a real upper tail: Table 3.1 rates")
    print("  S1 at light/sandy surface soils, so if neither source puts any")
    print("  meaningful share of cells above ~50% sand, no 250 m product")
    print("  resolves the terraces and the factor must be graded relative to")
    print("  the study area, with that limitation stated.")

    # does high sand sit on low ground, as alluvial terraces should?
    for r, d in sg.items():
        s = d["sg_sand"].ravel()
        s = s[np.isfinite(s)]
        hi = np.percentile(s, 95)
        print(f"\n  Region {r}: 95th percentile sand = {hi:.1f}%  "
              f"(cells above it: {int((s >= hi).sum()):,})")


if __name__ == "__main__":
    main()