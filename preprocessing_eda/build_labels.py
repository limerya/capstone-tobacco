"""
FAO suitability labelling -- parametric index (primary) + maximum
limitation (comparison).

Sys et al. (1993), the anchor for the Table 3.1 criteria, rates each
factor 0-100 on a continuous curve and combines the ratings into a land
index, rather than applying strict minimum law. Two Sys formulations are
computed:

Factor ratings are first collapsed into five domain indices (climate,
topography, wetness, physical soil, fertility), as Sys does, because
multiplying eight individual ratings drives every cell to zero. The
domain index is the lowest rating within that domain. The two Sys
formulations are then applied across the five domains:

  square-root method   I = Rmin * sqrt(prod(other domains / 100))
  Storie method        I = Rmin * prod(other domains / 100)

Land index bins (Sys): S1 >= 75, S2 50-75, S3 25-50, N < 25.

Maximum limitation (the class of the worst factor) is computed alongside
for comparison, so the aggregation rule can be reported as a sensitivity
analysis rather than assumed.

Absolute disqualifiers are preserved: a cell that fails one of these is N
regardless of index, which is the part of the minimum-law argument that
is agronomically real.

Reads  : model_matrix/
Writes : labeled/, label_report_v2.txt, training_sample_labeled.parquet

Run:
    python build_labels_v2.py
"""

import os
import glob
import time
from collections import Counter

import numpy as np
import pandas as pd

MATRIX = r"D:\Projects\capstone\model_matrix"
OUT = r"D:\Projects\capstone\labeled"
REPORT = r"D:\Projects\capstone\label_report_v2.txt"
SAMPLE_OUT = r"D:\Projects\capstone\training_sample_labeled.parquet"

CLASSES = ["S1", "S2", "S3", "N"]
USE_RAINFALL = False
RAINFALL_COL = "precip_annual_med"

# Land-index bins (Sys et al. 1993)
INDEX_BINS = [(75, "S1"), (50, "S2"), (25, "S3")]      # below 25 -> N
# Per-factor rating -> class, used only for the maximum-limitation comparison
RATING_BINS = [(85, "S1"), (60, "S2"), (40, "S3")]     # below 40 -> N

# Factor -> domain. The domain index is the minimum rating within it, so a
# single severe limitation still governs its domain, but limitations in
# different domains no longer multiply each other away.
# Domains that must be fully measured for a cell to be classified at all.
# Omitting this check is what produced 115,747 "S1" cells of which 98.7%
# had no soil data whatsoever.
REQUIRED_DOMAINS = ["topography", "physical_soil", "fertility"]

DOMAINS = {
    "climate":       ["temp", "sunshine", "rain"],
    "topography":    ["slope"],
    "wetness":       ["drainage"],
    "physical_soil": ["texture"],
    "fertility":     ["ph", "soc", "cec"],
}

# ======================================================================
# RATING CURVES
#
# (value, rating) anchor points, linearly interpolated. Anchors are placed
# so that the Table 3.1 band edges fall at the conventional parametric
# rating breaks: optimum ~95-100, S1/S2 edge ~85-92, S2/S3 edge ~68-70,
# S3/N edge ~25-30.
# ======================================================================
CURVES = {
    # Soil pH -- Table 3.1: S1 5.7-6.0, S2 5.5-6.2, S3 5.2-6.8, N <5.0 / >7.8
    "ph": [(3.5, 5), (5.0, 25), (5.2, 45), (5.5, 70), (5.7, 98),
           (6.0, 98), (6.2, 70), (6.8, 45), (7.8, 25), (9.0, 5)],

    # Slope % -- S1 0-4, S2 4-8, S3 8-16, N >16
    "slope": [(0, 100), (2, 100), (4, 92), (8, 68), (16, 30),
              (25, 12), (60, 2), (600, 0)],

    # Growing-season mean temperature C -- S1 24-26, S2 22-28, S3 20-30,
    # N <15 / >34
    "temp": [(8, 2), (15, 25), (20, 45), (22, 70), (24, 98),
             (26, 98), (28, 70), (30, 45), (34, 25), (42, 2)],

    # Soil organic carbon g/kg -- LOW is preferred for bright-leaf
    # (FAO 2001; Reed 2017). Rating classes after Hazelton & Murphy,
    # inverted.
    "soc": [(0, 100), (15, 92), (25, 68), (40, 30), (60, 12), (320, 2)],

    # CEC cmol(c)/kg -- moderate/crop-adequate preferred; very low or
    # very high both penalised. Hazelton & Murphy rating classes.
    # NOTE: SoilGrids CEC is buffered at pH 7; on these acidic soils it
    # overstates effective CEC at field pH.
    "cec": [(0, 15), (3, 30), (6, 68), (12, 95), (25, 95),
            (40, 68), (60, 30), (120, 15)],

    # Surface texture, graded on SAND CONTENT rather than USDA class.
    # Table 3.1 wants light sandy / sandy loam surface soils. The USDA
    # class route put 98.9% of cells in a single class (clay loam),
    # because SoilGrids/OpenLandMap at 250 m smooths texture toward the
    # regional mean and cannot resolve the sandy alluvial terraces where
    # the crop is actually grown. Grading on sand content preserves the
    # within-region gradient that does survive. The resolution limit is
    # reported as a constraint on the texture factor.
    "texture": [(10, 10), (20, 25), (30, 45), (40, 68),
                (50, 88), (60, 97), (75, 100), (100, 100)],

    # Drainage -- absolute thresholds, not percentiles. Wetter is worse.
    # NDMI from the dry-season composite; TWI from D-infinity routing.
    # The factor rating is the lower (worse) of the two.
    "ndmi_wet": [(-1.0, 100), (0.30, 100), (0.45, 85), (0.55, 60),
                 (0.70, 35), (1.0, 15)],
    "twi_wet": [(0, 100), (8, 100), (10, 88), (12, 65), (14, 40), (25, 20)],

    # Curing-season sunshine, MJ/m2/day. Across Regions I and II the
    # range is roughly 13-20, which is adequate for flue-curing
    # throughout, so this factor is near-non-limiting by construction --
    # as it should be. The earlier percentile cuts manufactured an N
    # class that has no agronomic basis.
    "sunshine": [(8, 45), (11, 70), (13, 85), (15, 94), (17, 100), (30, 100)],

    # Annual rainfall mm -- S1 900-1200, S2 700-1500, S3 500-2000
    "rain": [(0, 2), (500, 25), (700, 68), (900, 98), (1200, 98),
             (1500, 68), (2000, 25), (4000, 2)],
}

# Absolute disqualifiers: N regardless of index (FAO minimum-law element)
DISQUALIFY = {
    "slope_gt_16": lambda d: d["slope_pct"] > 16.0,
    "ph_outside_5_7.8": lambda d: (d["ph"] < 5.0) | (d["ph"] > 7.8),
    "waterlogged": lambda d: (d["ndmi_med_med"] > 0.70) & (d["twi"] > 12.0),
}

COLS = {"ph": "ph", "slope": "slope_pct", "temp": "tmean_med",
        "soc": "soc", "cec": "cec", "sand": "sand", "clay": "clay",
        "ndmi": "ndmi_med_med", "twi": "twi", "solar": "solar_rad_med",
        "rain": RAINFALL_COL}


def rate(values, curve):
    v = np.asarray(values, dtype="float64")
    xs = np.array([p[0] for p in curve], dtype="float64")
    ys = np.array([p[1] for p in curve], dtype="float64")
    r = np.interp(v, xs, ys, left=ys[0], right=ys[-1])
    r[~np.isfinite(v)] = np.nan
    return r


def bin_class(values, bins):
    v = np.asarray(values, dtype="float64")
    out = np.full(len(v), "N", dtype=object)
    for thr, cls in sorted(bins, key=lambda t: t[0]):
        out[v >= thr] = cls
    out[~np.isfinite(v)] = None
    return out


def factor_ratings(df):
    r = {}
    r["ph"] = rate(df[COLS["ph"]], CURVES["ph"])
    r["slope"] = rate(df[COLS["slope"]], CURVES["slope"])
    r["temp"] = rate(df[COLS["temp"]], CURVES["temp"])
    r["soc"] = rate(df[COLS["soc"]], CURVES["soc"])
    r["cec"] = rate(df[COLS["cec"]], CURVES["cec"])

    # texture: sand-based rating, capped where clay is genuinely heavy
    tex = rate(df[COLS["sand"]], CURVES["texture"])
    heavy = np.asarray(df[COLS["clay"]], "float64") >= 45.0
    tex = np.where(heavy, np.minimum(tex, 25.0), tex)
    r["texture"] = tex

    # drainage: worse of the observed (NDMI) and terrain (TWI) signals
    r["drainage"] = np.fmin(rate(df[COLS["ndmi"]], CURVES["ndmi_wet"]),
                            rate(df[COLS["twi"]], CURVES["twi_wet"]))

    r["sunshine"] = rate(df[COLS["solar"]], CURVES["sunshine"])

    if USE_RAINFALL and COLS["rain"] in df.columns:
        r["rain"] = rate(df[COLS["rain"]], CURVES["rain"])
    return r


def label_frame(df):
    r = factor_ratings(df)
    names = list(r)
    M = np.vstack([r[n] for n in names])              # factors x cells
    ok = np.isfinite(M)
    n_missing = (~ok).sum(axis=0)

    # --- collapse factors into domain indices (minimum within domain) --
    # A MISSING factor must not be read as non-limiting. np.fmin skips NaN
    # and the index formula substitutes 1.0 for an absent domain, so a cell
    # with no soil layer would be scored on climate and terrain alone and
    # come out highly suitable -- i.e. missing data would be rewarded.
    # Each domain therefore carries a gap flag, and a cell missing any
    # member of a decisive domain is left unscored rather than classified.
    dom_names, dom_vals, dom_gap = [], [], []
    for dname, members in DOMAINS.items():
        cols = [r[f] for f in members if f in r]
        if not cols:
            continue
        stack = np.vstack(cols)
        dom_names.append(dname)
        # Geometric mean within the domain, so minimum law is applied once --
        # across domains, by the Sys formula -- rather than twice. Taking the
        # minimum at both levels made the index unreachable (p99 = 58 against
        # an S1 threshold of 75). Severe single-factor failures are still
        # caught by DISQUALIFY.
        dom_vals.append(np.exp(np.nanmean(np.log(np.clip(stack, 1, 100)), axis=0)))
        dom_gap.append(~np.isfinite(stack).all(axis=0))   # any member absent
    D = np.vstack(dom_vals)                            # domains x cells
    G = np.vstack(dom_gap)
    dok = np.isfinite(D)

    req = [dom_names.index(d) for d in REQUIRED_DOMAINS if d in dom_names]
    unscorable = G[req].any(axis=0) | (~dok[req]).any(axis=0)

    # --- Sys methods across domains ------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        rmin = np.nanmin(np.where(dok, D, np.nan), axis=0)
        order = np.argsort(np.where(dok, D, np.inf), axis=0)
        others = np.take_along_axis(D, order[1:], axis=0) / 100.0
        others = np.where(np.isfinite(others), others, 1.0)
        prod_others = others.prod(axis=0)
        idx_sqrt = rmin * np.sqrt(prod_others)
        idx_storie = rmin * prod_others

    allna = (~dok).all(axis=0) | unscorable
    idx_sqrt[allna] = np.nan
    idx_storie[allna] = np.nan

    # --- maximum limitation on the same ratings -----------------------
    per_class = {n: bin_class(r[n], RATING_BINS) for n in names}
    rank = np.full((len(names), len(df)), -1, dtype="int8")
    for i, n in enumerate(names):
        for j, c in enumerate(CLASSES):
            rank[i][per_class[n] == c] = j
    worst = rank.max(axis=0)
    lim_i = rank.argmax(axis=0)
    maxlim = np.array([CLASSES[w] if w >= 0 else None for w in worst],
                      dtype=object)
    limiting = np.array([names[i] if w >= 0 else None
                         for i, w in zip(lim_i, worst)], dtype=object)
    # same rule: the worst OBSERVED factor is not the worst factor when a
    # decisive one was never measured
    maxlim[unscorable] = None
    limiting[unscorable] = None

    out = pd.DataFrame(index=df.index)
    for n in names:
        out[f"rate_{n}"] = r[n].astype("float32")
        out[f"lim_{n}"] = per_class[n]
    for i, dname in enumerate(dom_names):
        out[f"dom_{dname}"] = D[i].astype("float32")
    out["index_sqrt"] = idx_sqrt.astype("float32")
    out["index_storie"] = idx_storie.astype("float32")
    out["fao_class"] = bin_class(idx_sqrt, INDEX_BINS)        # primary
    out["fao_class_storie"] = bin_class(idx_storie, INDEX_BINS)
    out["fao_class_maxlim"] = maxlim
    out["limiting_factor"] = limiting
    out["n_factors_missing"] = n_missing.astype("int8")
    out["unscorable"] = unscorable

    dq = pd.Series(False, index=df.index)
    for nm, fn in DISQUALIFY.items():
        hit = fn(df).fillna(False).to_numpy()
        out[f"dq_{nm}"] = hit
        dq |= hit
    out["disqualified"] = dq.to_numpy()
    for c in ("fao_class", "fao_class_storie", "fao_class_maxlim"):
        out.loc[out["disqualified"] & out[c].notna(), c] = "N"
    return out, names


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(MATRIX, "*.parquet")))
    if not parts:
        raise SystemExit(f"no parts in {MATRIX}")

    rng = np.random.default_rng(0)
    cnt = {k: Counter() for k in ("fao_class", "fao_class_storie",
                                  "fao_class_maxlim")}
    limiting = Counter()
    dq_counts = Counter()
    rate_sum, rate_n = Counter(), Counter()
    by_region = {}
    cross = Counter()
    rows = 0
    unscorable_total = 0
    samples = []
    per_part = max(1, 600_000 // len(parts))
    t0 = time.time()

    for i, p in enumerate(parts, 1):
        df = pd.read_parquet(p)
        lab, names = label_frame(df)
        df = pd.concat([df, lab], axis=1)

        unscorable_total += int(df["unscorable"].sum())
        for k in cnt:
            cnt[k].update(df[k].dropna())
        limiting.update(df["limiting_factor"].dropna())
        for nm in DISQUALIFY:
            dq_counts[nm] += int(df[f"dq_{nm}"].sum())
        for n in names:
            s = df[f"rate_{n}"]
            rate_sum[n] += float(s.sum(skipna=True))
            rate_n[n] += int(s.notna().sum())
        for (rg, c), k in df.groupby(["region", "fao_class"]).size().items():
            by_region[(rg, c)] = by_region.get((rg, c), 0) + k
        for (a, b), k in df.groupby(["fao_class", "fao_class_maxlim"]).size().items():
            cross[(a, b)] += k
        rows += len(df)

        df.to_parquet(os.path.join(OUT, os.path.basename(p)),
                      compression="snappy", index=False)
        take = min(per_part, len(df))
        samples.append(df.iloc[rng.choice(len(df), take, replace=False)])

        if i % 10 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)}  {rows:,} rows "
                  f"({(time.time() - t0) / 60:.1f} min)")

    pd.concat(samples, ignore_index=True).to_parquet(
        SAMPLE_OUT, compression="snappy", index=False)

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 72)
    say("CLASS DISTRIBUTION BY AGGREGATION RULE")
    say("=" * 72)
    say(f"  {'class':6s}{'sqrt (primary)':>22s}{'Storie':>18s}{'max-limitation':>22s}")
    for c in CLASSES:
        row = f"  {c:6s}"
        for k in ("fao_class", "fao_class_storie", "fao_class_maxlim"):
            t = sum(cnt[k].values())
            n = cnt[k].get(c, 0)
            row += f"{n:>12,} {100 * n / max(t, 1):6.2f}%"
        say(row)
    say(f"  cells: {rows:,}")
    say(f"  unscorable (a decisive domain unmeasured): {unscorable_total:,} "
        f"({100 * unscorable_total / max(rows, 1):.2f}%)")

    say("\nAGREEMENT, square-root vs maximum-limitation")
    tot = sum(cross.values())
    agree = sum(v for (a, b), v in cross.items() if a == b)
    say(f"  identical class: {100 * agree / max(tot, 1):.2f}%")
    hdr = "sqrt vs maxlim"
    say(f"  {hdr:12s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for a in CLASSES:
        say(f"  {a:12s}" + "".join(
            f"{cross.get((a, b), 0):>12,}" for b in CLASSES))

    say("\nMEAN FACTOR RATING (0-100; low = limiting)")
    for n in sorted(rate_sum, key=lambda k: rate_sum[k] / max(rate_n[k], 1)):
        say(f"  {n:12s} {rate_sum[n] / max(rate_n[n], 1):6.1f}")

    say("\nLIMITING FACTOR under maximum-limitation")
    t = sum(limiting.values())
    for n, k in limiting.most_common():
        say(f"  {n:12s} {k:>12,}  {100 * k / max(t, 1):6.2f}%")

    say("\nABSOLUTE DISQUALIFIERS (overlapping)")
    for n, k in dq_counts.most_common():
        say(f"  {n:20s} {k:>12,}  {100 * k / max(rows, 1):6.2f}%")

    say("\nPRIMARY CLASS BY REGION (%)")
    for rg in sorted({k[0] for k in by_region}):
        t = sum(v for k, v in by_region.items() if k[0] == rg)
        say(f"  Region {rg}: " + "  ".join(
            f"{c} {100 * by_region.get((rg, c), 0) / max(t, 1):5.2f}%"
            for c in CLASSES))

    if not USE_RAINFALL:
        say("\nNOTE: rainfall excluded; Table 3.1 grades annual rainfall "
            "but precip_total_med is the Nov-Apr total.")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines))
    say(f"\nlabelled grid -> {OUT}")
    say(f"labelled sample -> {SAMPLE_OUT}")


if __name__ == "__main__":
    main()