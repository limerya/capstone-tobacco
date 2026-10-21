"""
Build the model-ready feature matrix.

  1. merges aligned/ with aligned_extras/ on (region, row, col)
  2. applies the cleaning decisions from the sanity checks
  3. keeps the reduced feature set (drops structurally redundant columns)
  4. assigns spatial block IDs for blocked cross-validation
  5. writes model_matrix/ plus a stratified sample for EDA and training
  6. prints a VIF table on the sample

Run:
    python finalize_matrix.py
"""

import os
import glob
import time

import numpy as np
import pandas as pd

ALIGNED = r"D:\Projects\capstone\aligned"
EXTRAS = r"D:\Projects\capstone\aligned_extras"
OUT = r"D:\Projects\capstone\model_matrix"
SAMPLE = r"D:\Projects\capstone\training_sample.parquet"

BLOCK_KM = 10          # spatial block size for blocked CV
SAMPLE_N = 600_000     # stratified sample size
WATER_CUT = 0.8        # water_freq_med above this = permanent water, dropped
MIN_SEASONS = 1        # drop cells with no usable satellite season

# ---------------------------------------------------------------- features
TERRAIN = ["elevation", "slope_pct", "northness", "eastness", "tpi", "twi"]
SOIL    = ["sand", "clay", "ph", "soc", "awc", "cec", "nitrogen"]
CLIMATE = ["precip_total_med", "precip_total_iqr", "precip_cv_med",
           "dry_spell_max_med", "tmean_med", "tmean_iqr", "vpd_med",
           "soil_moist_med", "solar_rad_med", "solar_rad_iqr"]
NDVI    = ["ndvi_med_med", "ndvi_med_iqr", "amplitude_med", "dss_peak_med"]
NDWI    = ["water_freq_med"]
NDMI    = ["ndmi_med_med", "ndmi_med_iqr", "ndmi_range_med", "stress_freq_med"]

FEATURES = TERRAIN + SOIL + CLIMATE + NDVI + NDWI + NDMI
KEYS = ["region", "province", "row", "col", "x", "y"]
QUALITY = ["ndvi_seasons_ok", "ndmi_seasons_ok"]

# dropped, with the reason, for the methodology write-up
DROPPED = {
    "slope":            "degrees; slope_pct carries the same information (r=0.99)",
    "silt":             "sand+clay+silt sums to 100; third fraction is determined",
    "drainage_proxy":   "exactly (sand-clay)*100",
    "tmin_med/tmax_med": "r>0.98 with tmean_med",
    "ndvi_p90/p10, peak_ndvi, min_ndvi, integral": "r>0.94 with ndvi_med_med",
    "ndwi_med/min/max/p90/range": "r>0.97 with NDVI bands; water_freq retained",
    "ndmi_min/p10":     "r>0.99 with ndmi_med_med",
    "sos_dss, eos_dss": "degenerate (84% zero / pinned to window edge)",
    "frost_free_days":  "not exported; tmin never below 14.5 C so non-limiting",
}


def clean(df):
    """Apply the fixes identified in the sanity checks."""
    n0 = len(df)

    # SRTM noise over water
    if "elevation" in df:
        df.loc[df["elevation"] < -5, "elevation"] = np.nan
        df["elevation"] = df["elevation"].clip(lower=0)

    # zero organic carbon is not physically possible on land
    if "soc" in df:
        df.loc[df["soc"] <= 0, "soc"] = np.nan

    # permanent water bodies are not candidate cropland
    if "water_freq_med" in df:
        df = df[df["water_freq_med"].fillna(0) < WATER_CUT]

    # cells with no usable satellite season
    if "ndvi_seasons_ok" in df:
        df = df[df["ndvi_seasons_ok"].fillna(0) >= MIN_SEASONS]

    return df, n0 - len(df)


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(ALIGNED, "*.parquet")))
    print(f"{len(parts)} windows to finalise")

    t0 = time.time()
    kept = dropped = 0
    frames = []
    rng = np.random.default_rng(0)
    per_part_sample = max(1, SAMPLE_N // max(len(parts), 1))

    for i, part in enumerate(parts, 1):
        name = os.path.basename(part)
        extra_path = os.path.join(EXTRAS, name)
        if not os.path.exists(extra_path):
            print(f"  WARNING no extras for {name}, skipping")
            continue

        base = pd.read_parquet(part)
        extra = pd.read_parquet(extra_path)

        df = base.merge(extra, on=["region", "row", "col"], how="left",
                        validate="one_to_one")

        df, n_drop = clean(df)
        dropped += n_drop

        # spatial blocks for blocked CV (x/y are metres in PRS92)
        step = BLOCK_KM * 1000
        df["block_id"] = (
            df["region"].astype(str) + "_"
            + (df["x"] // step).astype("int32").astype(str) + "_"
            + (df["y"] // step).astype("int32").astype(str)
        )

        cols = [c for c in KEYS + QUALITY + ["block_id"] + FEATURES
                if c in df.columns]
        missing = [c for c in FEATURES if c not in df.columns]
        if missing and i == 1:
            print(f"  NOTE missing feature columns: {missing}")

        df = df[cols]
        df.to_parquet(os.path.join(OUT, name), compression="snappy", index=False)
        kept += len(df)

        if len(df) > per_part_sample:
            frames.append(df.iloc[rng.choice(len(df), per_part_sample,
                                             replace=False)])
        else:
            frames.append(df)

        if i % 10 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)}  kept {kept:,}  "
                  f"({(time.time() - t0) / 60:.1f} min)")

    print(f"\nkept {kept:,} rows, dropped {dropped:,} "
          f"({100 * dropped / max(kept + dropped, 1):.2f}%)")
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "*.parquet")))
    print(f"model_matrix: {size / 1e9:.2f} GB")

    smp = pd.concat(frames, ignore_index=True)
    smp.to_parquet(SAMPLE, compression="snappy", index=False)
    print(f"training sample: {len(smp):,} rows -> {SAMPLE}")

    # ------------------------------------------------------------ VIF
    feats = [c for c in FEATURES if c in smp.columns]
    X = smp[feats].dropna()
    print(f"\nVIF on {len(X):,} complete rows, {len(feats)} features")
    C = np.corrcoef(X.to_numpy(dtype="float64").T)
    try:
        vif = np.diag(np.linalg.inv(C))
        order = np.argsort(-vif)
        print(f"  {'feature':24s} {'VIF':>9s}")
        for j in order:
            flag = "  <-- above 10" if vif[j] > 10 else ""
            print(f"  {feats[j]:24s} {vif[j]:9.2f}{flag}")
    except np.linalg.LinAlgError:
        print("  correlation matrix is singular: perfectly collinear features remain")

    print("\ncolumns dropped before VIF, with reasons:")
    for k, v in DROPPED.items():
        print(f"  {k:42s} {v}")


if __name__ == "__main__":
    main()