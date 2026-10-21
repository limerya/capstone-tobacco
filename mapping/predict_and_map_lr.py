"""
Wall-to-wall prediction and mapping from the baseline ordinal model.

Applies the fitted proportional-odds model to every labelled cell, writes
the predictions back as Parquet, renders the figures, and exports the
co-registered GeoTIFF stack that Section 3.4.1 promises.

Prediction is reimplemented from the saved coefficients rather than a
pickled statsmodels object, so it does not depend on library versions:
    xb          = X . beta
    P(y <= j)   = logistic(alpha_j - xb)
    P(y = j)    = P(y <= j) - P(y <= j-1)

Reads : labeled/ , model_spec.json , terrain_REGION_*.tif (for the grid)
Writes: predicted/            per-cell class + probabilities
        maps_output/*.png     figures
        maps_output/*.tif     GeoTIFFs (fao_class, predicted, index, limiting)
        maps_output/accuracy_by_province.csv

Run:
    python predict_and_map.py
"""

import os

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import glob
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import rasterio

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
SRC       = r"D:\Projects\capstone"
LABELED   = os.path.join(SRC, "labeled")
SPEC_PATH = os.path.join(SRC, "model_spec.json")
PRED_DIR  = os.path.join(SRC, "predicted")
OUT       = os.path.join(SRC, "maps_output")

TARGET_CRS, TARGET_RES = "EPSG:3121", 30.0
DISPLAY_STRIDE = 8          # ~240 m for the figures; GeoTIFFs stay at 30 m
DPI = 170

# S1 .. N, high to low suitability
CLASS_COLORS = {"S1": "#1a6b3c", "S2": "#7cb342", "S3": "#f0c419", "N": "#b0b7bd"}
CLASS_LABEL = {"S1": "S1 — highly suitable",
               "S2": "S2 — moderately suitable",
               "S3": "S3 — marginally suitable",
               "N":  "N — not suitable"}

# build_training_set.py renamed columns for the model; labeled/ still holds
# the originals. Map model variable -> source column, and rebuild `aspect`
# from the northness/eastness pair it was derived from.
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
    """Return a frame whose columns are the model's variable names."""
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


PART = re.compile(r"^region(?P<region>\d+)_r(?P<row>\d+)_c(?P<col>\d+)\.parquet$")


# -------------------------------------------------------------- model
def load_spec():
    with open(SPEC_PATH) as fh:
        s = json.load(fh)
    s["beta_vec"] = np.array([s["beta"][c] for c in s["final_cols"]], dtype="float64")
    s["alpha_vec"] = np.array(s["alphas"], dtype="float64")
    return s


def design(df, spec):
    """Rebuild the model matrix using the transforms fitted on train."""
    X = {}
    for v, cfg in spec["vars"].items():
        if cfg["kind"] == "circular":
            r = np.deg2rad(df[v].to_numpy(dtype="float64"))
            X[cfg["cols"][0]] = np.cos(r)
            X[cfg["cols"][1]] = np.sin(r)
        elif cfg["kind"] == "optimum":
            z = (df[v].to_numpy(dtype="float64") - cfg["center"]) / cfg["std"]
            X[cfg["cols"][0]] = z
            X[cfg["cols"][1]] = z ** 2
        else:
            x = df[v].to_numpy(dtype="float64")
            if cfg["log1p"]:
                x = np.log1p(np.clip(x, 0, None))
            X[cfg["cols"][0]] = (x - cfg["mean"]) / cfg["std"]
    return np.column_stack([X[c] for c in spec["final_cols"]])


def predict(Xm, spec):
    """Ordinal logit: class probabilities and the argmax class index."""
    xb = Xm @ spec["beta_vec"]
    a = spec["alpha_vec"]
    a = a[np.isfinite(a)]                       # drop the -inf / +inf sentinels
    cum = 1.0 / (1.0 + np.exp(-(a[None, :] - xb[:, None])))
    probs = np.diff(np.column_stack([np.zeros(len(xb)), cum,
                                     np.ones(len(xb))]), axis=1)
    return probs, probs.argmax(axis=1)


# -------------------------------------------------------------- grid
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


# -------------------------------------------------------------- figures
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


def cat_map(arr, names, title, path, stride=DISPLAY_STRIDE):
    a = arr[::stride, ::stride].astype("float32")
    a[a < 0] = np.nan
    cols = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(names), 2)))
    cmap = ListedColormap(cols[:len(names)])
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(a, origin="upper", cmap=cmap,
              norm=BoundaryNorm(np.arange(-0.5, len(names)), cmap.N))
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor=cols[i], label=n)
                       for i, n in enumerate(names)],
              loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


# -------------------------------------------------------------- main
def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    spec = load_spec()
    ORDER = spec["order"]
    cls_idx = {c: i for i, c in enumerate(ORDER)}
    tpl = templates()
    parts = sorted(glob.glob(os.path.join(LABELED, "*.parquet")))
    if not parts:
        raise SystemExit(f"no parts in {LABELED}")
    print(f"{len(parts)} parts | {len(spec['final_cols'])} model terms")

    need = sorted(spec["vars"].keys())
    wanted = [c for v in need for c in source_cols(v)] + \
            ["row", "col", "fao_class", "index_sqrt",
             "limiting_factor", "province"]
    have = set(pd.read_parquet(parts[0]).columns)
    USE_COLS = [c for c in dict.fromkeys(wanted) if c in have]
    missing = [c for c in dict.fromkeys(wanted) if c not in have]
    if missing:
        raise SystemExit(f"columns absent from {LABELED}: {missing}")
    print(f"  reading {len(USE_COLS)} source columns")

    lim_names = []
    grids = {r: {k: np.full((t["height"], t["width"]), -1, dtype="int8")
                 for k in ("true", "pred", "lim")} for r, t in tpl.items()}
    for r, t in tpl.items():
        grids[r]["index"] = np.full((t["height"], t["width"]), np.nan,
                                    dtype="float32")
        grids[r]["conf"] = np.full((t["height"], t["width"]), np.nan,
                                   dtype="float32")

    conf = np.zeros((len(ORDER), len(ORDER)), dtype="int64")
    prov_stats = {}
    n_pred = 0
    t0 = time.time()

    for i, p in enumerate(parts, 1):
        m = PART.match(os.path.basename(p))
        region = int(m.group("region"))
        df = to_model_names(pd.read_parquet(p, columns=USE_COLS), need)
        df = df[df["fao_class"].notna()]
        df = df.dropna(subset=need)
        if df.empty:
            continue

        Xm = design(df, spec)
        probs, k = predict(Xm, spec)
        pred = np.array([ORDER[j] for j in k], dtype=object)
        true_k = df["fao_class"].map(cls_idx).to_numpy()

        rr, cc = df["row"].to_numpy(), df["col"].to_numpy()
        g = grids[region]
        g["true"][rr, cc] = true_k.astype("int8")
        g["pred"][rr, cc] = k.astype("int8")
        g["index"][rr, cc] = df["index_sqrt"].to_numpy(dtype="float32")
        g["conf"][rr, cc] = probs.max(axis=1).astype("float32")

        lf = df["limiting_factor"].astype(str)
        for n in lf.unique():
            if n not in lim_names and n != "nan":
                lim_names.append(n)
        g["lim"][rr, cc] = lf.map(
            {n: j for j, n in enumerate(lim_names)}).fillna(-1).to_numpy()

        for a, b in zip(true_k, k):
            conf[a, b] += 1
        for prov, sub in df.assign(_p=pred).groupby("province", observed=True):
            s = prov_stats.setdefault(prov, [0, 0])
            s[0] += int((sub["_p"] == sub["fao_class"]).sum())
            s[1] += len(sub)

        out = pd.DataFrame({"region": region, "row": rr, "col": cc,
                            "fao_class": df["fao_class"].to_numpy(),
                            "predicted": pred,
                            "p_max": probs.max(axis=1).astype("float32")})
        for j, c in enumerate(ORDER):
            out[f"p_{c}"] = probs[:, j].astype("float32")
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
        class_map(g["true"], ORDER, f"FAO land suitability — {tag}\n"
                  f"parametric index (Sys square-root method)",
                  os.path.join(OUT, f"map_fao_class_R{r}.png"))
        class_map(g["pred"], ORDER, f"Ordinal model prediction — {tag}",
                  os.path.join(OUT, f"map_predicted_R{r}.png"))

        d = np.where((g["true"] >= 0) & (g["pred"] >= 0),
                     np.abs(g["true"].astype("int16") - g["pred"].astype("int16")),
                     np.nan).astype("float32")
        cont_map(d, f"Prediction error, classes apart — {tag}",
                 os.path.join(OUT, f"map_disagreement_R{r}.png"),
                 cmap="magma_r", vmin=0, vmax=2, label="classes apart")

        cont_map(g["index"], f"Land index (0–100) — {tag}",
                 os.path.join(OUT, f"map_land_index_R{r}.png"),
                 cmap="RdYlGn", vmin=0, vmax=80, label="land index")

        cont_map(g["conf"], f"Model confidence (max class probability) — {tag}",
                 os.path.join(OUT, f"map_confidence_R{r}.png"),
                 cmap="cividis", vmin=0.4, vmax=1.0, label="P(assigned class)")

        if lim_names:
            cat_map(g["lim"], lim_names,
                    f"Limiting factor (maximum-limitation rule) — {tag}",
                    os.path.join(OUT, f"map_limiting_factor_R{r}.png"))

        write_tif(np.where(g["true"] >= 0, g["true"], 255), tpl[r],
                  os.path.join(OUT, f"fao_class_R{r}.tif"), "uint8", 255)
        write_tif(np.where(g["pred"] >= 0, g["pred"], 255), tpl[r],
                  os.path.join(OUT, f"predicted_R{r}.tif"), "uint8", 255)
        write_tif(g["index"], tpl[r],
                  os.path.join(OUT, f"land_index_R{r}.tif"), "float32", np.nan)
        if lim_names:
            write_tif(np.where(g["lim"] >= 0, g["lim"], 255), tpl[r],
                      os.path.join(OUT, f"limiting_factor_R{r}.tif"), "uint8", 255)

    # ---------------------------------------------------------- report
    cm = pd.DataFrame(conf, index=[f"true_{c}" for c in ORDER],
                      columns=[f"pred_{c}" for c in ORDER])
    cm.to_csv(os.path.join(OUT, "wallwide_confusion.csv"))
    tot = conf.sum()
    exact = np.trace(conf)
    within1 = sum(conf[a, b] for a in range(len(ORDER))
                  for b in range(len(ORDER)) if abs(a - b) <= 1)

    rows = []
    for prov, (ok, n) in sorted(prov_stats.items(), key=lambda kv: -kv[1][1]):
        rows.append(dict(province=prov, cells=n, exact_pct=100*ok/max(n, 1)))
    prov_df = pd.DataFrame(rows)
    prov_df.to_csv(os.path.join(OUT, "accuracy_by_province.csv"), index=False)

    print("\n" + "=" * 62)
    print("WALL-TO-WALL PREDICTION")
    print("=" * 62)
    print(f"  cells predicted : {tot:,}")
    print(f"  exact agreement : {100*exact/tot:.2f}%")
    print(f"  within one class: {100*within1/tot:.2f}%")
    print("\nconfusion (rows = labelled class):")
    print(cm.to_string())
    print("\nclass shares:")
    for j, c in enumerate(ORDER):
        print(f"  {c:3s} labelled {100*conf[j].sum()/tot:6.2f}%   "
              f"predicted {100*conf[:, j].sum()/tot:6.2f}%   "
              f"area {conf[j].sum()*0.0009:,.0f} km2")
    print("\naccuracy by province:")
    print(prov_df.round(2).to_string(index=False))
    print(f"\nfigures and GeoTIFFs -> {OUT}")
    print(f"per-cell predictions -> {PRED_DIR}")


if __name__ == "__main__":
    main()