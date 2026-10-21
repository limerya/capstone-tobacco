"""
Build the model-ready training set from the labelled grid.

  1. class-stratified, block-aware sample (S1 taken in full, the rest capped)
  2. column names mapped to what feature_selection_ordinal_lr.py expects
  3. aspect reconstructed in degrees from northness/eastness, since that
     script does its own circular encoding
  4. sampling weights recorded so population proportions are recoverable
  5. block coverage reported per class -- the check that decides whether
     spatially blocked CV is even viable for the rare classes

Reads  : labeled/
Writes : feature_matrix.parquet      (the training set)
         feature_matrix_holdout.parquet (optional untouched blocks)
         priority_patch.py           (PRIORITY/SKEWED/OPTIMUM to paste in)
         training_set_report.txt

Run:
    python build_training_set.py
"""

import os
import glob
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

LABELED = r"D:\Projects\capstone\labeled"
OUT = r"D:\Projects\capstone\feature_matrix.parquet"
PATCH = r"D:\Projects\capstone\priority_patch.py"
REPORT = r"D:\Projects\capstone\training_set_report.txt"

TARGET_COL = "fao_class"          # square-root index (primary)
ALSO_KEEP = ["fao_class_maxlim", "fao_class_storie",
             "index_sqrt", "index_storie", "limiting_factor"]

CAP_PER_CLASS = 120_000           # S1 is taken whole; others capped here
DROP_INCOMPLETE = True            # ordinal LR needs complete cases
SEED = 42

# matrix column -> name expected by the feature-selection script
RENAME = {
    "ph": "soil_ph",
    "tmean_med": "temp_gs_mean",
    "ndmi_med_med": "ndmi_median",
    "sand": "sand_pct",
    "clay": "clay_pct",
    "nitrogen": "total_n",
    "solar_rad_med": "sunshine_rad",      # MJ/m2/day, not hours -- see note
    "ndvi_med_med": "ndvi_median",
    "water_freq_med": "water_freq",
    "precip_total_med": "rain_dryseason", # NOT annual; Table 3.1 grades annual
    "block_id": "spatial_block",
}

# everything carried through, in domain order
KEEP = [
    # topography
    "elevation", "slope_pct", "tpi", "twi", "northness", "eastness",
    # soil
    "sand", "clay", "ph", "soc", "awc", "cec", "nitrogen",
    # climate
    "precip_total_med", "precip_total_iqr", "precip_cv_med",
    "dry_spell_max_med", "tmean_med", "tmean_iqr", "vpd_med",
    "soil_moist_med", "solar_rad_med", "solar_rad_iqr",
    # land surface
    "ndvi_med_med", "ndvi_med_iqr", "amplitude_med", "dss_peak_med",
    "water_freq_med", "ndmi_med_med", "ndmi_med_iqr", "ndmi_range_med",
    "stress_freq_med",
]
KEYS = ["region", "province", "row", "col", "x", "y", "block_id"]

PRIORITY_PATCH = '''# Paste over the corresponding blocks in feature_selection_ordinal_lr.py.
# Names match feature_matrix.parquet as written by build_training_set.py.

DATA_PATH = "feature_matrix.parquet"
TARGET, BLOCK, REGION = "fao_class", "spatial_block", "region"
ORDER = ["N", "S3", "S2", "S1"]

PRIORITY = {
    # 1 = decisive (forced in), 2 = core Table 3.1, 3 = supporting,
    # 4 = supplementary
    "soil_ph": 1, "slope_pct": 1, "temp_gs_mean": 1, "ndmi_median": 1,
    "twi": 1,

    "sand_pct": 2, "clay_pct": 2, "soc": 2, "total_n": 2, "cec": 2,
    "sunshine_rad": 2, "rain_dryseason": 2,

    "elevation": 3, "aspect": 3, "vpd_med": 3, "ndvi_median": 3,
    "water_freq": 3, "awc": 3, "tpi": 3, "soil_moist_med": 3,
    "dry_spell_max_med": 3, "precip_cv_med": 3,

    "ndvi_med_iqr": 4, "ndmi_med_iqr": 4, "ndmi_range_med": 4,
    "stress_freq_med": 4, "amplitude_med": 4, "dss_peak_med": 4,
    "tmean_iqr": 4, "precip_total_iqr": 4, "solar_rad_iqr": 4,
}

CATEGORICAL = []                 # no land-cover layer was produced
CIRCULAR = ["aspect"]            # reconstructed from northness/eastness

# from the EDA skewness table (|skew| >= 1.0), restricted to columns here
SKEWED = ["soc", "total_n", "water_freq", "ndvi_med_iqr", "ndmi_med_iqr",
          "ndmi_range_med", "stress_freq_med", "elevation", "slope_pct",
          "twi", "amplitude_med"]

# Optimum-band variables: centred on the Table 3.1 S1 midpoint + squared term.
# rain_dryseason has NO entry: Table 3.1 grades ANNUAL rainfall and this
# column is the Nov-Apr total, so no published optimum applies to it.
OPTIMUM = {"soil_ph": 5.85, "temp_gs_mean": 25.0, "total_n": None,
           "cec": 18.5, "soc": None}
'''


def main():
    rng = np.random.default_rng(SEED)
    parts = sorted(glob.glob(os.path.join(LABELED, "*.parquet")))
    if not parts:
        raise SystemExit(f"no parts in {LABELED}")

    wanted = KEYS + KEEP + [TARGET_COL] + ALSO_KEEP
    available = set(pd.read_parquet(parts[0]).columns)
    use_cols = [c for c in wanted if c in available]
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"  note: not in the labelled parts, skipped: {missing}")

    pools = defaultdict(list)
    class_total = Counter()
    blocks_per_class = defaultdict(set)
    incomplete = 0
    rows = 0
    t0 = time.time()

    print(f"pass 1: scanning {len(parts)} parts")
    for i, p in enumerate(parts, 1):
        df = pd.read_parquet(p, columns=use_cols)
        df = df[df[TARGET_COL].notna()]
        rows += len(df)

        if DROP_INCOMPLETE:
            before = len(df)
            df = df.dropna(subset=[c for c in KEEP if c in df.columns])
            incomplete += before - len(df)

        for cls, g in df.groupby(TARGET_COL):
            class_total[cls] += len(g)
            blocks_per_class[cls].update(g["block_id"].unique())
            # reservoir-ish: keep a bounded random slice per part per class
            keep_n = min(len(g), max(2000, CAP_PER_CLASS // len(parts) * 4))
            pools[cls].append(g.iloc[rng.choice(len(g), keep_n, replace=False)]
                              if len(g) > keep_n else g)

        if i % 10 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)}  {rows:,} rows "
                  f"({(time.time() - t0) / 60:.1f} min)")

    frames = []
    frac = {}
    for cls, chunks in pools.items():
        pool = pd.concat(chunks, ignore_index=True)
        take = min(len(pool), CAP_PER_CLASS)
        sel = pool.iloc[rng.choice(len(pool), take, replace=False)] \
            if len(pool) > take else pool
        frac[cls] = len(sel) / max(class_total[cls], 1)
        sel = sel.copy()
        sel["sample_weight"] = 1.0 / max(frac[cls], 1e-12)
        frames.append(sel)

    df = pd.concat(frames, ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    # aspect in degrees, for the script's own sin/cos encoding
    df["aspect"] = (np.degrees(np.arctan2(df["eastness"], df["northness"]))
                    % 360.0).astype("float32")

    df["region"] = df["region"].map({1: "I", 2: "II"}).astype(str)
    df = df.rename(columns=RENAME)
    df.to_parquet(OUT, compression="snappy", index=False)

    with open(PATCH, "w") as fh:
        fh.write(PRIORITY_PATCH)

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 70)
    say("TRAINING SET")
    say("=" * 70)
    say(f"  rows: {len(df):,}   columns: {df.shape[1]}")
    if DROP_INCOMPLETE:
        say(f"  dropped for incomplete predictors: {incomplete:,} "
            f"({100 * incomplete / max(rows, 1):.2f}% of labelled cells)")

    say("\n  class      population        sampled    frac    weight")
    for c in ["S1", "S2", "S3", "N"]:
        n_pop = class_total.get(c, 0)
        n_s = int((df[TARGET_COL] == c).sum())
        say(f"  {c:4s} {n_pop:>14,} {n_s:>14,} {frac.get(c, 0):8.4f} "
            f"{1 / max(frac.get(c, 1e-12), 1e-12):9.1f}")

    say("\n  BLOCK COVERAGE (spatially blocked CV needs blocks, not just cells)")
    for c in ["S1", "S2", "S3", "N"]:
        nb = len(blocks_per_class.get(c, ()))
        flag = "  <-- too few for 5-fold blocked CV" if nb < 25 else ""
        say(f"  {c:4s} {nb:>6,} distinct 10 km blocks{flag}")

    say("\n  BY REGION")
    for r, g in df.groupby("region"):
        vc = g[TARGET_COL].value_counts()
        say(f"  Region {r}: " + "  ".join(
            f"{c} {vc.get(c, 0):>7,}" for c in ["S1", "S2", "S3", "N"]))

    say("\nNOTES FOR THE WRITE-UP")
    say("  - Classes are oversampled relative to the population. For the")
    say("    ordinal logit this shifts the intercepts (cut-points) but not")
    say("    the slopes, so odds ratios remain interpretable; report the")
    say("    population proportions alongside.")
    say("  - sample_weight holds the inverse sampling fraction if you need")
    say("    population-level estimates.")
    say("  - solar_rad_med is renamed sunshine_rad (MJ/m2/day). Table 3.1")
    say("    says 'sunshine hours'; the units differ and the text needs")
    say("    updating to match.")
    say("  - rain_dryseason is the Nov-Apr total, not the annual rainfall")
    say("    Table 3.1 grades. It carries no OPTIMUM entry for that reason.")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines))
    say(f"\ntraining set -> {OUT}")
    say(f"PRIORITY patch -> {PATCH}")


if __name__ == "__main__":
    main()