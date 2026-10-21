"""
Wall-to-wall prediction and mapping — ANN (multilayer perceptron).

Fourth of the mapping scripts, matching predict_and_map.py (ordinal LR),
predict_and_map_rf.py (random forest) and predict_and_map_xgb.py exactly in
conventions: same class colours, same filenames, same layer set, same tables,
same cell selection. Every output is prefixed `ann_`, every title says "ANN".

  ordinal LR        random forest     xgboost           ann
  ---------------   ---------------   ---------------   ---------------
  predicted/        predicted_rf/     predicted_xgb/    predicted_ann/
  maps_output/      maps_output_rf/   maps_output_xgb/  maps_output_ann/

Builds a FOUR-WAY consensus when the other prediction folders exist.

Two things differ from the tree scripts, by necessity rather than style:
  - the TRAINING StandardScaler is loaded and applied; refitting on the full
    grid would leak the holdout distribution into the map
  - class order needs no reordering: the ANN was trained on integer labels
    0 = N .. 3 = S1, so softmax columns already follow ORDER (the forest's
    predict_proba columns were alphabetical and had to be reordered)

Reads : labeled/ , nn_outputs/ann_model.keras , nn_outputs/nn_preprocessing.joblib ,
        terrain_REGION_*.tif
Writes: predicted_ann/ , maps_output_ann/

Run (standalone process, not a notebook cell — the PROJ fix must precede
the rasterio import):
    python predict_and_map_ann.py
"""

import os

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import glob
import re
import time

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

# ----------------------------------------------------------------- CONFIG
SRC        = r"D:\Projects\capstone"
LABELED    = os.path.join(SRC, "labeled")
MODEL_PATH = os.path.join(SRC, "nn_outputs", "ann_model.keras")
PREP_PATH  = os.path.join(SRC, "nn_outputs", "nn_preprocessing.joblib")
PRED_DIR   = os.path.join(SRC, "predicted_ann")
OUT        = os.path.join(SRC, "maps_output_ann")
LR_DIR     = os.path.join(SRC, "predicted")        # ordinal LR, for comparison
RF_DIR     = os.path.join(SRC, "predicted_rf")     # random forest
XGB_DIR    = os.path.join(SRC, "predicted_xgb")    # xgboost

MODEL_TAG = "ANN"
FILE_TAG  = "ann"

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


def consensus_map(arr, title, path, n_models, stride=DISPLAY_STRIDE):
    """1 = every available model agrees, 0 = they do not."""
    a = arr[::stride, ::stride].astype("float32")
    a[a < 0] = np.nan
    cmap = ListedColormap(["#c0392b", "#dfe6e9"])
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(a, origin="upper", cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5], 2))
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor="#dfe6e9",
                             label=f"all {n_models} models agree"),
                       Patch(facecolor="#c0392b", label="models disagree")],
              loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[{(time.time()-t0)/60:6.1f} min] {m}", flush=True)

    for p in (MODEL_PATH, PREP_PATH):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run Stage 12 of the notebook first")

    model = keras.models.load_model(MODEL_PATH)
    prep = joblib.load(PREP_PATH)
    FEATURES, ORDER, scaler = prep["features"], prep["order"], prep["scaler"]
    cls_idx = {c: i for i, c in enumerate(ORDER)}
    level = np.arange(len(ORDER), dtype="float64")   # 0 = N .. 3 = S1

    # The ANN was trained on integer codes 0..3, so softmax columns already
    # follow ORDER. Assert the width rather than assuming.
    assert model.output_shape[-1] == len(ORDER), (
        f"model emits {model.output_shape[-1]} classes, expected {len(ORDER)}")
    assert model.input_shape[-1] == len(FEATURES), (
        f"model takes {model.input_shape[-1]} inputs, "
        f"preprocessing lists {len(FEATURES)}")

    mean = scaler.mean_.astype("float64")
    sd = np.sqrt(scaler.var_).astype("float64")

    print(f"{MODEL_TAG}: {len(FEATURES)} features, "
          f"{model.count_params():,} parameters")
    print("features:", FEATURES)
    print("scaler  : training-set statistics loaded (never refitted here)")

    tpl = templates()
    parts = sorted(glob.glob(os.path.join(LABELED, "*.parquet")))
    if not parts:
        raise SystemExit(f"no parts in {LABELED}")

    wanted = [c for v in FEATURES for c in source_cols(v)] + \
             ["row", "col", "fao_class", "index_sqrt", "limiting_factor",
              "province"]
    have = set(pd.read_parquet(parts[0]).columns)
    USE_COLS = [c for c in dict.fromkeys(wanted) if c in have]
    missing = [c for c in dict.fromkeys(wanted)
               if c not in have and c in [x for v in FEATURES
                                          for x in source_cols(v)]]
    if missing:
        raise SystemExit(f"columns absent from {LABELED}: {missing}")
    print(f"reading {len(USE_COLS)} source columns from {len(parts)} parts")

    grids = {}
    prov_stats = {}
    rows_out = []

    for i, path in enumerate(parts, 1):
        m = re.search(r"region[_]?(\d)", os.path.basename(path), re.I)
        r = int(m.group(1)) if m else int(
            pd.read_parquet(path, columns=["region"])["region"].iloc[0])
        if r not in tpl:
            continue
        if r not in grids:
            t = tpl[r]
            grids[r] = {
                "true":  np.full((t["height"], t["width"]), -1, dtype="int8"),
                "pred":  np.full((t["height"], t["width"]), -1, dtype="int8"),
                "conf":  np.full((t["height"], t["width"]), np.nan, dtype="float32"),
                "level": np.full((t["height"], t["width"]), np.nan, dtype="float32"),
            }
        g = grids[r]

        raw = pd.read_parquet(path, columns=USE_COLS)
        df = to_model_names(raw, FEATURES)
        for c in FEATURES:
            a = df[c].to_numpy(dtype="float64")
            a[~np.isfinite(a)] = np.nan
            df[c] = a
        # same cell selection as the other three scripts
        df = df[df["fao_class"].notna()].dropna(subset=FEATURES)
        if df.empty:
            continue

        Xm = ((df[FEATURES].to_numpy(dtype="float64") - mean) / sd
              ).astype("float32")
        P = np.empty((len(Xm), len(ORDER)), dtype="float32")
        for s in range(0, len(Xm), CHUNK):
            P[s:s + CHUNK] = model.predict(Xm[s:s + CHUNK],
                                           batch_size=8192, verbose=0)
        k = P.argmax(axis=1)

        rr = df["row"].to_numpy()
        cc = df["col"].to_numpy()
        true_k = df["fao_class"].map(cls_idx).to_numpy()

        g["true"][rr, cc] = true_k.astype("int8")
        g["pred"][rr, cc] = k.astype("int8")
        g["conf"][rr, cc] = P.max(axis=1)
        g["level"][rr, cc] = (P @ level).astype("float32")

        if "province" in df.columns:
            sub = df.assign(_p=[ORDER[j] for j in k])
            for prov, s2 in sub.groupby("province"):
                s = prov_stats.setdefault(prov, [0, 0])
                s[0] += int((s2["_p"] == s2["fao_class"]).sum())
                s[1] += len(s2)

        rows_out.append(pd.DataFrame({
            "region": r, "row": rr, "col": cc,
            "pred_code": k.astype("int8"),
            f"{FILE_TAG}_predicted": [ORDER[j] for j in k],
            "fao_class": df["fao_class"].to_numpy(),
            "confidence": P.max(axis=1),
            "expected_level": (P @ level).astype("float32"),
        }))
        if i % 10 == 0 or i == len(parts):
            tick(f"{i}/{len(parts)} parts")

    if rows_out:
        allp = pd.concat(rows_out, ignore_index=True)
        for r, sub in allp.groupby("region"):
            sub.to_parquet(os.path.join(
                PRED_DIR, f"{FILE_TAG}_pred_region{r}.parquet"), index=False)
        del rows_out

    # ------------------------------------------------------------- figures
    pre = os.path.join(OUT, f"{FILE_TAG}_")
    for r, g in grids.items():
        tag = f"Region {'I' if r == 1 else 'II'}"
        class_map(g["true"], ORDER,
                  f"FAO reference labelling — {tag}",
                  f"{pre}map_fao_class_R{r}.png")
        class_map(g["pred"], ORDER, f"{MODEL_TAG} prediction — {tag}",
                  f"{pre}map_predicted_R{r}.png")

        d = np.where((g["true"] >= 0) & (g["pred"] >= 0),
                     np.abs(g["true"].astype("int16") - g["pred"].astype("int16")),
                     np.nan).astype("float32")
        cont_map(d, f"{MODEL_TAG} disagreement with reference — {tag}",
                 f"{pre}map_disagreement_R{r}.png", cmap="magma_r",
                 vmin=0, vmax=3, label="classes apart")
        cont_map(g["level"], f"{MODEL_TAG} expected suitability level — {tag}",
                 f"{pre}map_expected_level_R{r}.png", cmap="YlGn",
                 vmin=0, vmax=3, label="0 = N  …  3 = S1")
        cont_map(g["conf"], f"{MODEL_TAG} prediction confidence — {tag}",
                 f"{pre}map_confidence_R{r}.png", cmap="cividis",
                 vmin=0.25, vmax=1.0, label="max class probability")

        write_tif(np.where(g["pred"] >= 0, g["pred"], 255), tpl[r],
                  f"{pre}predicted_R{r}.tif", "uint8", 255)
        write_tif(g["level"], tpl[r], f"{pre}expected_level_R{r}.tif",
                  "float32", np.nan)
        write_tif(g["conf"], tpl[r], f"{pre}confidence_R{r}.tif",
                  "float32", np.nan)
        tick(f"region {r} figures and rasters written")

    # ------------------------------------------------- wall-wide agreement
    cm = pd.DataFrame(0, index=[f"true_{c}" for c in ORDER],
                      columns=[f"pred_{c}" for c in ORDER])
    for g in grids.values():
        ok = (g["true"] >= 0) & (g["pred"] >= 0)
        t, p = g["true"][ok], g["pred"][ok]
        for a in range(len(ORDER)):
            for b in range(len(ORDER)):
                cm.iloc[a, b] += int(np.sum((t == a) & (p == b)))
    cm.to_csv(os.path.join(OUT, f"{FILE_TAG}_wallwide_confusion.csv"))
    tot = cm.to_numpy().sum()
    print(f"\nwall-wide agreement with the reference labelling: "
          f"{np.trace(cm.to_numpy()) / max(tot, 1):.4%} of {tot:,} cells")
    print(cm.to_string())

    if prov_stats:
        pv = pd.DataFrame([{"province": k, "correct": v[0], "n": v[1],
                            "accuracy": v[0] / max(v[1], 1)}
                           for k, v in prov_stats.items()]
                          ).sort_values("accuracy")
        pv.to_csv(os.path.join(OUT, f"{FILE_TAG}_accuracy_by_province.csv"),
                  index=False)
        print("\nagreement by province (worst five)")
        print(pv.head().round(4).to_string(index=False))

    # ------------------------------------------------ cross-model comparison
    others = {"lr": LR_DIR, "rf": RF_DIR, "xgb": XGB_DIR}
    loaded = {}
    mine = pd.concat(
        [pd.read_parquet(p, columns=["region", "row", "col", "pred_code"])
         for p in sorted(glob.glob(os.path.join(PRED_DIR, "*.parquet")))],
        ignore_index=True) if glob.glob(os.path.join(PRED_DIR, "*.parquet")) \
        else pd.DataFrame()

    for name, d in others.items():
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not files or mine.empty:
            continue
        frames = []
        for f in files:
            cols = pd.read_parquet(f).head(0).columns
            code = next((c for c in ("pred_code", "pred", "prediction")
                         if c in cols), None)
            if code is None or "row" not in cols:
                continue
            frames.append(pd.read_parquet(f, columns=["region", "row", "col", code])
                          .rename(columns={code: name}))
        if frames:
            loaded[name] = pd.concat(frames, ignore_index=True)

    if loaded:
        J = mine.rename(columns={"pred_code": FILE_TAG})
        for name, t in loaded.items():
            J = J.merge(t, on=["region", "row", "col"], how="inner")
        if len(J):
            for name in loaded:
                agree = float((J[FILE_TAG] == J[name]).mean())
                pd.crosstab(
                    pd.Categorical.from_codes(J[FILE_TAG].to_numpy().astype(int),
                                              categories=ORDER),
                    pd.Categorical.from_codes(J[name].to_numpy().astype(int),
                                              categories=ORDER),
                    rownames=[FILE_TAG], colnames=[name]
                ).to_csv(os.path.join(OUT, f"{FILE_TAG}_vs_{name}_crosstab.csv"))
                print(f"\n{MODEL_TAG} vs {name.upper()}: {agree:.4%} agreement "
                      f"on {len(J):,} shared cells")

            names = [FILE_TAG] + list(loaded)
            same = np.ones(len(J), dtype=bool)
            for name in loaded:
                same &= (J[FILE_TAG].to_numpy() == J[name].to_numpy())
            print(f"all {len(names)} models agree on {same.mean():.2%} of cells")

            for r, sub in J.assign(_same=same.astype("int8")).groupby("region"):
                c = np.full((tpl[r]["height"], tpl[r]["width"]), -1, dtype="int8")
                c[sub["row"].to_numpy(), sub["col"].to_numpy()] = \
                    sub["_same"].to_numpy()
                consensus_map(c, f"Model consensus ({', '.join(names)}) — "
                                 f"Region {'I' if r == 1 else 'II'}",
                              os.path.join(OUT, f"{FILE_TAG}_map_consensus_R{r}.png"),
                              len(names))
                write_tif(np.where(c >= 0, c, 255), tpl[r],
                          os.path.join(OUT, f"{FILE_TAG}_consensus_R{r}.tif"),
                          "uint8", 255)
                del c
            pd.DataFrame([{"models": ",".join(names), "n": len(J),
                           "unanimous": float(same.mean())}]).to_csv(
                os.path.join(OUT, f"{FILE_TAG}_consensus_summary.csv"), index=False)

    tick(f"done -> {OUT}")


if __name__ == "__main__":
    main()