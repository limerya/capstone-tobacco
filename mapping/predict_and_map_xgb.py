"""
Wall-to-wall prediction and mapping — XGBOOST model.

Third of the mapping scripts, alongside predict_and_map.py (ordinal LR)
and predict_and_map_rf.py (random forest). Every output here is prefixed
`xgb_` and every figure title says "XGBoost".

  ordinal LR            random forest          xgboost
  -------------------   --------------------   ----------------------
  predicted/            predicted_rf/          predicted_xgb/
  maps_output/          maps_output_rf/        maps_output_xgb/

Also builds a THREE-WAY comparison when the other two prediction folders
exist: agreement rates, a pairwise cross-tab, and a consensus map showing
where all three models agree — a model-uncertainty surface that none of
the individual maps provides.

Note on class order: XGBoost was trained on integer labels 0 = N … 3 = S1,
so predict_proba columns are already in suitability order. (The forest's
were alphabetical and had to be reordered.)

Reads : labeled/ , xgb_outputs/xgb_model_final.joblib , terrain_REGION_*.tif
Writes: predicted_xgb/ , maps_output_xgb/

Run:
    python predict_and_map_xgb.py
"""

import os

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import glob
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
import joblib

_db = glob.glob(os.path.join(os.path.dirname(rasterio.__file__), "**", "proj.db"),
                recursive=True)
if _db:
    os.environ["PROJ_DATA"] = os.path.dirname(_db[0])
    os.environ["PROJ_LIB"] = os.path.dirname(_db[0])

from rasterio.warp import calculate_default_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ----------------------------------------------------------------- CONFIG
SRC        = r"D:\Projects\capstone"
LABELED    = os.path.join(SRC, "labeled")
MODEL_PATH = os.path.join(SRC, "xgb_outputs", "xgb_model_final.joblib")
PRED_DIR   = os.path.join(SRC, "predicted_xgb")
OUT        = os.path.join(SRC, "maps_output_xgb")
LR_DIR     = os.path.join(SRC, "predicted")        # ordinal LR, for comparison
RF_DIR     = os.path.join(SRC, "predicted_rf")     # random forest

MODEL_TAG = "XGBoost"
FILE_TAG  = "xgb"

TARGET_CRS, TARGET_RES = "EPSG:3121", 30.0
DISPLAY_STRIDE = 8          # ~240 m for figures; GeoTIFFs stay at 30 m
DPI = 170
CHUNK = 500_000             # rows per predict call

CLASS_COLORS = {"S1": "#1a6b3c", "S2": "#7cb342", "S3": "#f0c419", "N": "#b0b7bd"}
CLASS_LABEL = {"S1": "S1 — highly suitable", "S2": "S2 — moderately suitable",
               "S3": "S3 — marginally suitable", "N": "N — not suitable"}

# model variable -> column name in labeled/
SOURCE = {
    "soil_ph": "ph", "temp_gs_mean": "tmean_med", "ndmi_median": "ndmi_med_med",
    "sand_pct": "sand", "clay_pct": "clay", "total_n": "nitrogen",
    "sunshine_rad": "solar_rad_med", "ndvi_median": "ndvi_med_med",
    "water_freq": "water_freq_med", "rain_dryseason": "precip_total_med",
}
DERIVED = {"aspect": ["northness", "eastness"]}

PART = re.compile(r"^region(?P<region>\d+)_r(?P<row>\d+)_c(?P<col>\d+)\.parquet$")


def source_cols(v):
    return DERIVED.get(v, [SOURCE.get(v, v)])


def to_model_names(df, need):
    out = pd.DataFrame(index=df.index)
    for v in need:
        if v in DERIVED:
            out[v] = (np.degrees(np.arctan2(df["eastness"], df["northness"]))
                      % 360.0)
        else:
            out[v] = df[SOURCE.get(v, v)]
    for extra in ("row", "col", "fao_class", "index_sqrt",
                  "limiting_factor", "province"):
        if extra in df.columns:
            out[extra] = df[extra]
    return out


def templates():
    out = {}
    for p in glob.glob(os.path.join(SRC, "**", "terrain_REGION_*.tif"),
                       recursive=True):
        if "aligned" in p.split(os.sep) or "_clean" in p:
            continue
        r = int(re.search(r"REGION_(\d)", p).group(1))
        with rasterio.open(p) as src:
            tr, w, h = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height, *src.bounds,
                resolution=TARGET_RES)
        out[r] = {"transform": tr, "width": w, "height": h}
    return out


def write_tif(arr, tpl, path, dtype, nodata):
    with rasterio.open(path, "w", driver="GTiff", height=tpl["height"],
                       width=tpl["width"], count=1, dtype=dtype,
                       crs=TARGET_CRS, transform=tpl["transform"],
                       nodata=nodata, compress="deflate", tiled=True) as dst:
        dst.write(arr.astype(dtype), 1)


def class_map(arr, order, title, path, stride=DISPLAY_STRIDE):
    a = arr[::stride, ::stride].astype("float32")
    a[a < 0] = np.nan
    cmap = ListedColormap([CLASS_COLORS[c] for c in order])
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(a, origin="upper", cmap=cmap,
              norm=BoundaryNorm(np.arange(-0.5, len(order)), cmap.N))
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor=CLASS_COLORS[c], label=CLASS_LABEL[c])
                       for c in order],
              loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def cont_map(arr, title, path, cmap="viridis", stride=DISPLAY_STRIDE,
             vmin=None, vmax=None, label=""):
    a = arr[::stride, ::stride].astype("float32")
    a[~np.isfinite(a)] = np.nan
    fig, ax = plt.subplots(figsize=(7, 10))
    if vmin is None:
        vmin, vmax = np.nanpercentile(a, [2, 98])
    im = ax.imshow(a, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.65, label=label)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def consensus_map(arr, title, path, stride=DISPLAY_STRIDE):
    """1 = all three models agree, 0 = they do not."""
    a = arr[::stride, ::stride].astype("float32")
    a[a < 0] = np.nan
    cmap = ListedColormap(["#c0392b", "#dfe6e9"])
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(a, origin="upper", cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5], 2))
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor="#dfe6e9", label="all three models agree"),
                       Patch(facecolor="#c0392b", label="models disagree")],
              loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    model, FEATURES, ORDER = bundle["model"], bundle["features"], bundle["order"]
    params = bundle.get("params", {})
    try:
        model.set_params(n_jobs=-1)
    except Exception:
        pass
    cls_idx = {c: i for i, c in enumerate(ORDER)}
    level = np.arange(len(ORDER), dtype="float64")   # 0 = N .. 3 = S1

    # XGBoost was trained on integer codes 0..3, so predict_proba columns
    # already follow ORDER. Assert it rather than assuming.
    classes = [int(c) for c in getattr(model, "classes_", range(len(ORDER)))]
    assert classes == list(range(len(ORDER))), (
        f"unexpected class order {classes}; expected 0..{len(ORDER)-1}")

    n_rounds = params.get("n_estimators", "?")
    print(f"{MODEL_TAG}: {len(FEATURES)} features, {n_rounds} rounds "
          f"(~{n_rounds * len(ORDER) if isinstance(n_rounds, int) else '?'} trees)")
    print("features:", FEATURES)

    tpl = templates()
    parts = sorted(glob.glob(os.path.join(LABELED, "*.parquet")))
    if not parts:
        raise SystemExit(f"no parts in {LABELED}")

    wanted = [c for v in FEATURES for c in source_cols(v)] + \
             ["row", "col", "fao_class", "index_sqrt", "limiting_factor",
              "province"]
    have = set(pd.read_parquet(parts[0]).columns)
    USE_COLS = [c for c in dict.fromkeys(wanted) if c in have]
    missing = [c for c in dict.fromkeys(wanted) if c not in have]
    if missing:
        raise SystemExit(f"columns absent from {LABELED}: {missing}")
    print(f"reading {len(USE_COLS)} source columns from {len(parts)} parts")

    grids = {}
    for r, t in tpl.items():
        grids[r] = {
            "true":  np.full((t["height"], t["width"]), -1, dtype="int8"),
            "pred":  np.full((t["height"], t["width"]), -1, dtype="int8"),
            "conf":  np.full((t["height"], t["width"]), np.nan, dtype="float32"),
            "level": np.full((t["height"], t["width"]), np.nan, dtype="float32"),
        }

    conf_mat = np.zeros((len(ORDER), len(ORDER)), dtype="int64")
    prov_stats = {}
    n_pred = 0
    t0 = time.time()

    for i, p in enumerate(parts, 1):
        m = PART.match(os.path.basename(p))
        region = int(m.group("region"))

        df = to_model_names(pd.read_parquet(p, columns=USE_COLS), FEATURES)
        df = df[df["fao_class"].notna()].dropna(subset=FEATURES)
        if df.empty:
            continue

        Xm = df[FEATURES].to_numpy(dtype="float32")
        P = np.empty((len(Xm), len(ORDER)), dtype="float32")
        for s in range(0, len(Xm), CHUNK):
            P[s:s + CHUNK] = model.predict_proba(Xm[s:s + CHUNK])

        k = P.argmax(axis=1)
        pred = np.array([ORDER[j] for j in k], dtype=object)
        true_k = df["fao_class"].map(cls_idx).to_numpy()

        rr, cc = df["row"].to_numpy(), df["col"].to_numpy()
        g = grids[region]
        g["true"][rr, cc] = true_k.astype("int8")
        g["pred"][rr, cc] = k.astype("int8")
        g["conf"][rr, cc] = P.max(axis=1)
        g["level"][rr, cc] = (P @ level).astype("float32")

        for a, b in zip(true_k, k):
            conf_mat[a, b] += 1
        for prov, sub in df.assign(_p=pred).groupby("province", observed=True):
            s = prov_stats.setdefault(prov, [0, 0])
            s[0] += int((sub["_p"] == sub["fao_class"]).sum())
            s[1] += len(sub)

        out = pd.DataFrame({"region": region, "row": rr, "col": cc,
                            "fao_class": df["fao_class"].to_numpy(),
                            "xgb_predicted": pred,
                            "xgb_p_max": P.max(axis=1),
                            "xgb_level": (P @ level).astype("float32")})
        for j, c in enumerate(ORDER):
            out[f"xgb_p_{c}"] = P[:, j]
        out.to_parquet(os.path.join(PRED_DIR, os.path.basename(p)),
                       compression="snappy", index=False)
        n_pred += len(out)

        if i % 10 == 0 or i == len(parts):
            print(f"  {i}/{len(parts)}  {n_pred:,} cells "
                  f"({(time.time()-t0)/60:.1f} min)")

    # ---------------------------------------------------------- figures
    print("\nrendering")
    for r, g in grids.items():
        tag = f"Region {'I' if r == 1 else 'II'}"
        pre = os.path.join(OUT, f"{FILE_TAG}_")

        class_map(g["true"], ORDER,
                  f"FAO land suitability (labelled) — {tag}\n"
                  f"parametric index, Sys square-root method",
                  f"{pre}map_fao_class_R{r}.png")
        class_map(g["pred"], ORDER, f"{MODEL_TAG} prediction — {tag}",
                  f"{pre}map_predicted_R{r}.png")

        d = np.where((g["true"] >= 0) & (g["pred"] >= 0),
                     np.abs(g["true"].astype("int16") - g["pred"].astype("int16")),
                     np.nan).astype("float32")
        cont_map(d, f"{MODEL_TAG} error, classes apart — {tag}",
                 f"{pre}map_disagreement_R{r}.png",
                 cmap="magma_r", vmin=0, vmax=2, label="classes apart")

        cont_map(g["level"],
                 f"{MODEL_TAG} expected suitability level — {tag}\n"
                 f"0 = N, 1 = S3, 2 = S2, 3 = S1",
                 f"{pre}map_expected_level_R{r}.png",
                 cmap="RdYlGn", vmin=0, vmax=3, label="expected class")

        cont_map(g["conf"],
                 f"{MODEL_TAG} confidence (max class probability) — {tag}",
                 f"{pre}map_confidence_R{r}.png",
                 cmap="cividis", vmin=0.4, vmax=1.0, label="P(assigned class)")

        write_tif(np.where(g["pred"] >= 0, g["pred"], 255), tpl[r],
                  f"{pre}predicted_R{r}.tif", "uint8", 255)
        write_tif(g["level"], tpl[r], f"{pre}expected_level_R{r}.tif",
                  "float32", np.nan)
        write_tif(g["conf"], tpl[r], f"{pre}confidence_R{r}.tif",
                  "float32", np.nan)

    # ------------------------------------------- three-way comparison
    lr_ok = os.path.isdir(LR_DIR) and glob.glob(os.path.join(LR_DIR, "*.parquet"))
    rf_ok = os.path.isdir(RF_DIR) and glob.glob(os.path.join(RF_DIR, "*.parquet"))
    if lr_ok or rf_ok:
        print("\ncomparing against the other models")
        cons = {r: np.full((t["height"], t["width"]), -1, dtype="int8")
                for r, t in tpl.items()}
        agree = Counter()
        pair = Counter()

        for p in sorted(glob.glob(os.path.join(PRED_DIR, "*.parquet"))):
            name = os.path.basename(p)
            x = pd.read_parquet(p, columns=["region", "row", "col",
                                            "fao_class", "xgb_predicted"])
            j = x
            if lr_ok and os.path.exists(os.path.join(LR_DIR, name)):
                lr = pd.read_parquet(os.path.join(LR_DIR, name),
                                     columns=["region", "row", "col", "predicted"])
                j = j.merge(lr.rename(columns={"predicted": "lr_predicted"}),
                            on=["region", "row", "col"], how="inner")
            if rf_ok and os.path.exists(os.path.join(RF_DIR, name)):
                rf = pd.read_parquet(os.path.join(RF_DIR, name),
                                     columns=["region", "row", "col",
                                              "rf_predicted"])
                j = j.merge(rf, on=["region", "row", "col"], how="inner")
            if j.empty:
                continue

            cols = [c for c in ("lr_predicted", "rf_predicted", "xgb_predicted")
                    if c in j.columns]
            agree["total"] += len(j)
            for c in cols:
                agree[c.replace("_predicted", "_correct")] += \
                    int((j[c] == j["fao_class"]).sum())
            if len(cols) == 3:
                same = ((j["lr_predicted"] == j["rf_predicted"]) &
                        (j["rf_predicted"] == j["xgb_predicted"])).to_numpy()
                agree["all_three"] += int(same.sum())
                reg = j["region"].to_numpy()
                rr, cc = j["row"].to_numpy(), j["col"].to_numpy()
                for r in cons:
                    m = reg == r
                    if m.any():
                        cons[r][rr[m], cc[m]] = same[m].astype("int8")
            if "rf_predicted" in j.columns:
                for (a, b), n in j.groupby(["rf_predicted",
                                            "xgb_predicted"]).size().items():
                    pair[(a, b)] += n

        t = max(agree["total"], 1)
        print(f"  cells compared        : {agree['total']:,}")
        for k_, lbl in (("lr_correct", "ordinal LR"),
                        ("rf_correct", "random forest"),
                        ("xgb_correct", "XGBoost")):
            if agree[k_]:
                print(f"  {lbl:<20s}: {100*agree[k_]/t:6.2f}% correct")
        if agree["all_three"]:
            print(f"  all three agree       : {100*agree['all_three']/t:6.2f}%")

        if pair:
            cm2 = pd.DataFrame(
                [[pair.get((a, b), 0) for b in ORDER] for a in ORDER],
                index=[f"RF_{c}" for c in ORDER],
                columns=[f"XGB_{c}" for c in ORDER])
            cm2.to_csv(os.path.join(OUT, f"{FILE_TAG}_vs_rf_crosstab.csv"))
            print("\n  random forest (rows) vs XGBoost (columns):")
            print(cm2.to_string())

        for r, c in cons.items():
            if (c >= 0).any():
                tag = f"Region {'I' if r == 1 else 'II'}"
                consensus_map(c, f"Model consensus — {tag}\n"
                                 f"ordinal LR, random forest and XGBoost",
                              os.path.join(OUT, f"{FILE_TAG}_map_consensus_R{r}.png"))
                write_tif(np.where(c >= 0, c, 255), tpl[r],
                          os.path.join(OUT, f"{FILE_TAG}_consensus_R{r}.tif"),
                          "uint8", 255)

    # ---------------------------------------------------------- report
    cm = pd.DataFrame(conf_mat, index=[f"true_{c}" for c in ORDER],
                      columns=[f"pred_{c}" for c in ORDER])
    cm.to_csv(os.path.join(OUT, f"{FILE_TAG}_wallwide_confusion.csv"))
    tot = conf_mat.sum()
    exact = np.trace(conf_mat)
    within1 = sum(conf_mat[a, b] for a in range(len(ORDER))
                  for b in range(len(ORDER)) if abs(a - b) <= 1)
    far = tot - within1

    prov_df = pd.DataFrame(
        [dict(province=k, cells=n, exact_pct=100*ok/max(n, 1))
         for k, (ok, n) in sorted(prov_stats.items(), key=lambda kv: -kv[1][1])])
    prov_df.to_csv(os.path.join(OUT, f"{FILE_TAG}_accuracy_by_province.csv"),
                   index=False)

    print("\n" + "=" * 64)
    print(f"WALL-TO-WALL PREDICTION — {MODEL_TAG.upper()}")
    print("=" * 64)
    print(f"  cells predicted  : {tot:,}")
    print(f"  exact agreement  : {100*exact/tot:.3f}%")
    print(f"  within one class : {100*within1/tot:.3f}%")
    print(f"  two classes apart: {far:,} cells ({100*far/tot:.4f}%)")
    print("\nconfusion (rows = labelled class):")
    print(cm.to_string())
    print("\nclass shares:")
    for j, c in enumerate(ORDER):
        print(f"  {c:3s} labelled {100*conf_mat[j].sum()/tot:6.2f}%   "
              f"predicted {100*conf_mat[:, j].sum()/tot:6.2f}%   "
              f"area {conf_mat[j].sum()*0.0009:,.0f} km2")
    print("\naccuracy by province:")
    print(prov_df.round(2).to_string(index=False))
    print(f"\nfigures and GeoTIFFs -> {OUT}  (all prefixed '{FILE_TAG}_')")
    print(f"per-cell predictions -> {PRED_DIR}")


if __name__ == "__main__":
    main()