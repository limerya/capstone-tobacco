"""
Sanity checks on the aligned feature matrix.

Streams every Parquet part for exact counts / ranges / missingness, then
draws a random sample for percentiles and correlations, so it never tries
to hold 45 million rows in memory.

Run:
    python sanity_check.py "D:\\Projects\\capstone\\aligned"
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

SAMPLE_PER_PART = 4000          # rows sampled per part for percentiles/corr
CELL_AREA_KM2 = 0.0009          # 30 m x 30 m

# Physically plausible ranges after unit conversion.
EXPECTED = {
    "elevation":        (-10, 3000),
    "slope":            (0, 90),
    "slope_pct":        (0, 600),
    "northness":        (-1, 1),
    "eastness":         (-1, 1),
    "sand":             (0, 100),
    "clay":             (0, 100),
    "silt":             (0, 100),
    "ph":               (3.0, 9.0),
    "soc":              (0, 400),
    "awc":              (0, 100),
    "tmean_med":        (10, 35),
    "tmin_med":         (5, 30),
    "tmax_med":         (15, 42),
    "vpd_med":          (0, 5),
    "precip_total_med": (0, 3000),
    "precip_cv_med":    (0, 15),
    "soil_moist_med":   (0, 600),
    "ndvi_med_med":     (-1, 1),
    "peak_ndvi_med":    (-1, 1),
    "ndwi_med_med":     (-1, 1),
    "water_freq_med":   (0, 1),
    "ndmi_med_med":     (-1, 1),
    "stress_freq_med":  (0, 1),
}

# Table 3.1 factors and the column that carries each.
TABLE31 = {
    "Soil pH":                 "ph",
    "Slope (%)":               "slope_pct",
    "Growing-season temp":     "tmean_med",
    "Rainfall":                "precip_total_med",
    "Drainage (NDMI)":         "ndmi_med_med",
    "Surface texture (sand)":  "sand",
    "Surface texture (clay)":  "clay",
    "Soil organic carbon":     "soc",
    "CEC":                     "cec",
    "Frost-free days":         "frost_free_days",
    "Sunshine hours":          "sunshine_hours",
}


def stream_stats(parts):
    n = 0
    agg = {}
    regions = {}
    provinces = {}
    dup_keys = set()
    dup_found = 0

    for i, p in enumerate(parts, 1):
        df = pd.read_parquet(p)
        n += len(df)

        for r, c in df["region"].value_counts().items():
            regions[r] = regions.get(r, 0) + c
        for pr, c in df["province"].value_counts().items():
            if c:
                provinces[pr] = provinces.get(pr, 0) + c

        key = set(zip(df["region"].to_numpy(),
                      df["row"].to_numpy(), df["col"].to_numpy()))
        before = len(dup_keys)
        dup_keys |= key
        # rows in this part that landed on a cell already seen
        dup_found += len(df) - (len(dup_keys) - before)

        for c in df.columns:
            if df[c].dtype.kind not in "fiu":
                continue
            v = df[c].to_numpy(dtype="float64")
            ok = np.isfinite(v)
            s = agg.setdefault(c, {"n": 0, "nan": 0, "min": np.inf,
                                   "max": -np.inf, "sum": 0.0, "zero": 0})
            s["n"] += v.size
            s["nan"] += int((~ok).sum())
            if ok.any():
                s["min"] = min(s["min"], float(v[ok].min()))
                s["max"] = max(s["max"], float(v[ok].max()))
                s["sum"] += float(v[ok].sum())
                s["zero"] += int((v[ok] == 0).sum())

        if i % 20 == 0:
            print(f"  scanned {i}/{len(parts)} parts, {n:,} rows")

    return n, agg, regions, provinces, dup_found


def draw_sample(parts, rng):
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        if len(df) > SAMPLE_PER_PART:
            df = df.iloc[rng.choice(len(df), SAMPLE_PER_PART, replace=False)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else r"D:\Projects\capstone\aligned"
    parts = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not parts:
        sys.exit(f"no parquet parts in {folder}")

    size_gb = sum(os.path.getsize(p) for p in parts) / 1e9
    print(f"{len(parts)} parts, {size_gb:.2f} GB on disk\n")

    print("streaming exact stats...")
    n, agg, regions, provinces, dups = stream_stats(parts)

    print("\n" + "=" * 78)
    print("1. SHAPE AND COVERAGE")
    print("=" * 78)
    print(f"rows: {n:,}   columns: {len(agg) + 2}")
    for r in sorted(regions):
        km2 = regions[r] * CELL_AREA_KM2
        print(f"  Region {r}: {regions[r]:,} cells = {km2:,.0f} km2")
    print("  reference land area: Region I ~12,840 km2, "
          "Region II ~26,800 km2 excluding Batanes")
    print(f"  duplicate (region,row,col) keys: {dups:,}"
          + ("  <-- CHECK" if dups else "  ok"))

    print("\nprovince breakdown:")
    for pr, c in sorted(provinces.items(), key=lambda kv: -kv[1]):
        print(f"  {pr:20s} {c:>12,}  ({c * CELL_AREA_KM2:>8,.0f} km2)")

    print("\n" + "=" * 78)
    print("2. MISSINGNESS (top 20)")
    print("=" * 78)
    miss = sorted(((s["nan"] / s["n"] * 100, c) for c, s in agg.items()),
                  reverse=True)
    for pct, c in miss[:20]:
        flag = "  <-- CHECK" if pct > 25 else ""
        print(f"  {c:26s} {pct:6.2f}%{flag}")
    print(f"  ({sum(1 for pct, _ in miss if pct == 0)} columns fully complete)")

    print("\n" + "=" * 78)
    print("3. VALUE RANGES vs EXPECTED")
    print("=" * 78)
    print(f"  {'column':26s} {'min':>10s} {'mean':>10s} {'max':>10s}   verdict")
    for c, (lo, hi) in EXPECTED.items():
        if c not in agg:
            print(f"  {c:26s} {'--':>10s} {'--':>10s} {'--':>10s}   MISSING COLUMN")
            continue
        s = agg[c]
        valid = s["n"] - s["nan"]
        mean = s["sum"] / valid if valid else np.nan
        bad = (s["min"] < lo) or (s["max"] > hi)
        print(f"  {c:26s} {s['min']:10.3f} {mean:10.3f} {s['max']:10.3f}   "
              + ("OUT OF RANGE  <-- CHECK" if bad else "ok"))

    print("\n" + "=" * 78)
    print("4. TABLE 3.1 FACTOR COVERAGE")
    print("=" * 78)
    for factor, col in TABLE31.items():
        print(f"  {factor:26s} {'present' if col in agg else 'ABSENT - needs export'}"
              f"  ({col})")

    print("\nsampling for percentiles and correlations...")
    rng = np.random.default_rng(0)
    smp = draw_sample(parts, rng)
    print(f"sample: {len(smp):,} rows")

    print("\n" + "=" * 78)
    print("5. KEY DISTRIBUTIONS (sample percentiles)")
    print("=" * 78)
    keys = [c for c in ["elevation", "slope_pct", "ph", "sand", "clay", "soc",
                        "tmean_med", "precip_total_med", "ndvi_med_med",
                        "ndmi_med_med", "water_freq_med", "ndvi_med_iqr"]
            if c in smp.columns]
    print(smp[keys].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(3))

    print("\n" + "=" * 78)
    print("6. OCEAN / MASK LEAKAGE")
    print("=" * 78)
    if "water_freq_med" in smp:
        wet = (smp["water_freq_med"] > 0.8).mean() * 100
        print(f"  cells with water_freq > 0.8: {wet:.2f}%   "
              + ("<-- CHECK, sea may still be in the mask" if wet > 5 else "ok"))
    if "ndvi_med_med" in smp:
        neg = (smp["ndvi_med_med"] < 0).mean() * 100
        print(f"  cells with negative median NDVI: {neg:.2f}%   "
              + ("<-- CHECK" if neg > 10 else "ok"))
    if "elevation" in smp:
        below = (smp["elevation"] < -5).mean() * 100
        print(f"  cells below -5 m elevation: {below:.2f}%")

    print("\n" + "=" * 78)
    print("7. SEASON AVAILABILITY")
    print("=" * 78)
    for c in [c for c in smp.columns if c.endswith("seasons_ok")]:
        vc = smp[c].value_counts(normalize=True).sort_index() * 100
        print(f"  {c}:")
        for k, v in vc.items():
            print(f"      {int(k)} usable seasons: {v:5.1f}%")

    print("\n" + "=" * 78)
    print("8. REDUNDANCY (|r| > 0.9 in the sample)")
    print("=" * 78)
    num = smp.select_dtypes("number").drop(
        columns=[c for c in ("row", "col", "x", "y", "region") if c in smp], errors="ignore")
    corr = num.corr(numeric_only=True).abs()
    seen = set()
    hits = []
    for a in corr.columns:
        for b in corr.columns:
            if a >= b or (a, b) in seen:
                continue
            seen.add((a, b))
            r = corr.loc[a, b]
            if np.isfinite(r) and r > 0.9:
                hits.append((r, a, b))
    for r, a, b in sorted(hits, reverse=True)[:25]:
        print(f"  {r:5.3f}  {a}  ~  {b}")
    print(f"  {len(hits)} pairs above 0.9 - candidates to drop before VIF")

    if {"sand", "clay", "silt"} <= set(smp.columns):
        tot = smp[["sand", "clay", "silt"]].sum(axis=1)
        print(f"\n  sand+clay+silt: mean {tot.mean():.1f}, sd {tot.std():.2f}")
        print("  (near-constant sum means silt is redundant - drop one fraction)")


if __name__ == "__main__":
    main()