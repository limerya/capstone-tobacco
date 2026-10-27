"""
Check that the lean align_stack.py reproduces the existing aligned/ parts.

The existing aligned/ was patched in place by patch_climate.py, so it holds
the repaired climate values. The lean align_stack.py applies the same joint
climate rule at source. Every column the lean run writes must therefore be
bit-identical to the same column in aligned/. Columns present only in
aligned/ are the ones the lean run skips on purpose.

Steps:
    1. python align_stack.py "D:\\Projects\\capstone"     (writes aligned_lean/)
    2. python verify_lean_align.py "D:\\Projects\\capstone" aligned aligned_lean
       Checks every part in aligned_lean/ against aligned/.
    3. Only after PASS: replace aligned/ with aligned_lean/.

Quick smoke test first (optional): set ONLY in align_stack.py to a few part
names, run steps 1-2, then set ONLY back to set() and delete aligned_lean/.

Writes nothing. Exit code 0 = PASS.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

# columns finalize_matrix.py reads from aligned/ (everything else in
# model_matrix comes from aligned_extras/ or is derived there)
NEEDED = [
    "region", "province", "row", "col", "x", "y",
    "ndvi_seasons_ok", "ndmi_seasons_ok",
    "elevation", "slope_pct", "northness", "eastness", "tpi",
    "sand", "clay", "ph", "soc", "awc",
    "precip_total_med", "precip_total_iqr", "precip_cv_med",
    "dry_spell_max_med", "tmean_med", "tmean_iqr", "vpd_med", "soil_moist_med",
    "ndvi_med_med", "ndvi_med_iqr", "amplitude_med", "dss_peak_med",
    "water_freq_med",
    "ndmi_med_med", "ndmi_med_iqr", "ndmi_range_med", "stress_freq_med",
]


def identical(a, b):
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        x, y = a.to_numpy(), b.to_numpy()
        if x.dtype != y.dtype:
            return False, f"dtype {x.dtype} vs {y.dtype}"
        if np.array_equal(x, y, equal_nan=True):
            return True, ""
        both = np.isfinite(x) & np.isfinite(y)
        nan_diff = int((np.isnan(x) != np.isnan(y)).sum())
        maxd = float(np.max(np.abs(x[both] - y[both]))) if both.any() else 0.0
        return False, f"max |diff| {maxd:.3g}, NaN-pattern diffs {nan_diff}"
    same = (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all()
    return bool(same), "" if same else "values differ"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\Projects\capstone"
    ref_dir = os.path.join(root, sys.argv[2] if len(sys.argv) > 2 else "aligned")
    new_dir = os.path.join(root, sys.argv[3] if len(sys.argv) > 3 else "aligned_lean")

    new_parts = sorted(glob.glob(os.path.join(new_dir, "*.parquet")))
    if not new_parts:
        sys.exit(f"no parts in {new_dir} -- run align_stack.py --out first")

    ok_all = True
    skipped = set()
    for p in new_parts:
        name = os.path.basename(p)
        ref_path = os.path.join(ref_dir, name)
        if not os.path.exists(ref_path):
            print(f"{name}: no matching part in {ref_dir}")
            ok_all = False
            continue
        new = pd.read_parquet(p).sort_values(["row", "col"]).reset_index(drop=True)
        ref = pd.read_parquet(ref_path).sort_values(["row", "col"]).reset_index(drop=True)

        print(f"\n{name}: {len(new):,} rows (reference {len(ref):,})")
        if len(new) != len(ref):
            print("  FAIL row count differs")
            ok_all = False
            continue

        missing = [c for c in NEEDED if c not in new.columns]
        if missing:
            print(f"  FAIL lean output lacks columns finalize_matrix needs: {missing}")
            ok_all = False

        bad = 0
        for c in new.columns:
            if c not in ref.columns:
                print(f"  FAIL {c}: not in reference")
                ok_all = False
                bad += 1
                continue
            same, why = identical(new[c], ref[c])
            if not same:
                print(f"  FAIL {c}: {why}")
                ok_all = False
                bad += 1
        skipped |= set(ref.columns) - set(new.columns)
        if not bad:
            print(f"  all {new.shape[1]} columns identical")

    print(f"\ncolumns skipped by the lean run ({len(skipped)}): "
          + ", ".join(sorted(skipped)))
    print("\nPASS" if ok_all else "\nFAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()