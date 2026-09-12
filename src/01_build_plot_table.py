"""
01_build_plot_table.py
======================
Turns the raw inputs into one clean table per flight, one row per scored plot.

    raw zonal stats  +  disease scores  +  plot-to-variety lookup  +  texture stats
                                        |
                                        v
    data/processed/A2_clean_tex_norm.csv   (Flight 2, training)
    data/processed/A1_clean_tex_norm.csv   (Flight 1, temporally independent test)

Run from the repo root:
    python src/01_build_plot_table.py

What happens, step by step
--------------------------
1. Zonal statistics. Each flight file has several rows per plot. They are
   collapsed to one row per plot with max(), which keeps the one non-empty value
   each index column has for that plot. Grid rows labelled 'G' are dropped.
2. Disease scores. The 1 to 10 visual score for each plot, per flight.
3. Varieties. Two lookups are stacked: the main conventional block, and the
   15-variety sub-trial (two replicate plots each). A spelling variant of one
   variety name is merged so its replicates share a group (the correction is
   read from variety_spelling_fixes.csv, stored with the data, so no variety
   names appear in this code). That matters: the
   variety is the grouping unit in cross-validation, so a misspelling would
   silently split a variety across train and test.
4. Join, then drop plots with no disease score.
5. Texture. Three band statistics are added (NIR max, RedEdge 90th percentile,
   Red min). Flight 1 reflectance sits on a different numeric scale from Flight 2,
   so each texture column is z-scored WITHIN its own flight (mean 0, SD 1).

Every expected value in the sanity check was printed by an executed cell of the
thesis notebook (my_code.html); the cell number sits beside each check.
The tables are only written if every check passes (or if --force is given).
If processed tables already exist, the rebuilt ones are compared with them first
and nothing is overwritten on any difference.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, PROCESSED_DIR, require  # noqa: E402

# ---------------------------------------------------------------------------
# Input files (names as used in the thesis; place them in DATA_DIR)
# ---------------------------------------------------------------------------
STATS_FILE = {"1": "ALL_stats_1_JB_jack_data.csv", "2": "ALL_stats_2_JB_jack_data.csv"}
SCORES_FILE = "final_disease_data.csv"
VARIETY_FILE = "variety_and_plot_IDs.xlsx"
TEXTURE_FILE = "texture_features_selected.csv"

INDICES = ["canopy_cover", "NDVI", "RDVI", "GNDVI", "NDRE", "PVR",
           "EVI2", "TVI", "TCARI", "OSAVI", "TCARI_OSAVI"]
TEXTURE_COLS = ["NIR_max", "RedEdge_p90", "Red_min"]
SPELLING_FIXES_FILE = "variety_spelling_fixes.csv"   # columns: wrong,right (kept with the data, not in code)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def load_zonal_stats(flight):
    df = pd.read_csv(require(DATA_DIR / STATS_FILE[flight]))
    df = df[df["Disease_Plot_ID"].astype(str) != "G"].copy()
    df["Disease_Plot_ID"] = df["Disease_Plot_ID"].astype(str)
    cols = [f"{i}_{flight}_{s}" for i in INDICES for s in ["mean", "median", "stdev"]
            if not (i == "canopy_cover" and s == "median")]
    g = df.groupby("Disease_Plot_ID")[cols].max().reset_index()
    g.columns = ["Disease_Plot_ID"] + [c.replace(f"_{flight}_", "_") for c in cols]
    return g


def load_scores():
    dis = pd.read_csv(require(DATA_DIR / SCORES_FILE))
    dis["Disease_Plot_ID"] = dis["Disease_Plot_ID"].astype(str)
    return dis


def load_varieties():
    path = require(DATA_DIR / VARIETY_FILE)

    conv = pd.read_excel(path, sheet_name="2024 Conv Block Labels", header=1)
    fixes = pd.read_csv(require(DATA_DIR / SPELLING_FIXES_FILE))
    conv["Variety"] = conv["Variety"].replace(dict(zip(fixes["wrong"], fixes["right"])))
    vm = conv[["Variety", "2024 Plot No"]].dropna().copy()
    vm["Disease_Plot_ID"] = pd.to_numeric(vm["2024 Plot No"], errors="coerce").astype("Int64").astype(str)
    vm = vm.rename(columns={"Variety": "Variety_ID"})[["Disease_Plot_ID", "Variety_ID"]]

    m = pd.read_excel(path, sheet_name="2024 Map", header=None)
    sub = m.iloc[1:, [33, 34, 35]].copy()          # variety, block 1 plot, block 2 plot
    sub.columns = ["Variety", "Block1_PlotID", "Block2_PlotID"]
    sub = sub.dropna(subset=["Variety"])
    long1 = sub[["Variety", "Block1_PlotID"]].rename(columns={"Block1_PlotID": "Disease_Plot_ID"})
    long2 = sub[["Variety", "Block2_PlotID"]].rename(columns={"Block2_PlotID": "Disease_Plot_ID"})
    vm_extra = pd.concat([long1, long2], ignore_index=True)
    vm_extra["Disease_Plot_ID"] = vm_extra["Disease_Plot_ID"].astype(int).astype(str)
    vm_extra = vm_extra.rename(columns={"Variety": "Variety_ID"})

    return vm, vm_extra, pd.concat([vm, vm_extra], ignore_index=True)


def build_flight(flight, stats, dis, vm_full):
    scores = dis[dis["Flight"] == f"Flight {flight}"][["Disease_Plot_ID", "Disease_class"]]
    full = stats.merge(scores, on="Disease_Plot_ID", how="left")
    full = full.merge(vm_full, on="Disease_Plot_ID", how="left")
    clean = full.dropna(subset=["Disease_class"]).copy()
    return full, clean


def add_texture(clean, texture, flight):
    t = texture[["Disease_Plot_ID", f"NIR_{flight}_max", f"RedEdge_{flight}_p90", f"Red_{flight}_min"]].copy()
    t.columns = ["Disease_Plot_ID"] + TEXTURE_COLS
    out = clean.merge(t, on="Disease_Plot_ID", how="left")
    for col in TEXTURE_COLS:                       # z-score within this flight only
        out[col + "_norm"] = (out[col] - out[col].mean()) / out[col].std()
    return out


def class_counts(df):
    s = df["Disease_class"]
    return (int((s <= 6).sum()), int(((s >= 7) & (s <= 9)).sum()), int((s == 10).sum()))


def score_counts(df):
    vc = df["Disease_class"].value_counts()
    return {k: int(vc.get(float(k), 0)) for k in range(1, 11)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="write outputs even if a sanity check fails (debugging only)")
    args = ap.parse_args()

    F2, F1 = load_zonal_stats("2"), load_zonal_stats("1")
    dis = load_scores()
    vm, vm_extra, vm_full = load_varieties()
    A2, A2_clean = build_flight("2", F2, dis, vm_full)
    A1, A1_clean = build_flight("1", F1, dis, vm_full)

    texture = pd.read_csv(require(DATA_DIR / TEXTURE_FILE))
    texture_shape = texture.shape
    texture["Disease_Plot_ID"] = texture["Disease_Plot_ID"].astype(str)
    A2_tex = add_texture(A2_clean, texture, "2")
    A1_tex = add_texture(A1_clean, texture, "1")
    norm_cols = [c + "_norm" for c in TEXTURE_COLS]

    # -----------------------------------------------------------------------
    # SANITY CHECK: every expected value was printed by a cell in my_code.html
    # -----------------------------------------------------------------------
    checks = [
        ("cell 16",  "Flight 2 zonal stats shape", F2.shape, (572, 33)),
        ("cell 41",  "Flight 1 zonal stats shape", F1.shape, (572, 33)),
        ("cell 17",  "disease score table shape", dis.shape, (1144, 3)),
        ("cell 24",  "main-block lookup shape", vm.shape, (542, 2)),
        ("cell 24",  "main-block varieties after spelling fix", vm["Variety_ID"].nunique(), 271),
        ("cell 26",  "sub-trial lookup shape", vm_extra.shape, (30, 2)),
        ("cell 27",  "combined lookup unique plot IDs", vm_full["Disease_Plot_ID"].nunique(), 572),
        ("cell 27",  "combined lookup unique varieties", vm_full["Variety_ID"].nunique(), 286),
        ("cell 30",  "imagery plots with no variety", len(set(F2["Disease_Plot_ID"]) - set(vm_full["Disease_Plot_ID"])), 0),
        ("cell 37",  "Flight 2 joined shape", A2.shape, (572, 35)),
        ("cell 37",  "Flight 2 plots with no variety", int(A2["Variety_ID"].isna().sum()), 0),
        ("cell 37",  "Flight 2 plots with no score", int(A2["Disease_class"].isna().sum()), 45),
        ("cell 38",  "Flight 2 scored plots shape", A2_clean.shape, (527, 35)),
        ("cell 38",  "Flight 2 diseased / moderate / healthy", class_counts(A2_clean), (86, 47, 394)),
        ("cell 21",  "Flight 2 plots per raw score 1-10", score_counts(A2_clean),
         {1: 54, 2: 2, 3: 4, 4: 15, 5: 7, 6: 4, 7: 16, 8: 30, 9: 1, 10: 394}),
        ("cell 42",  "Flight 1 joined shape", A1.shape, (572, 35)),
        ("cell 42",  "Flight 1 plots with no variety", int(A1["Variety_ID"].isna().sum()), 0),
        ("cell 42",  "Flight 1 plots with no score", int(A1["Disease_class"].isna().sum()), 7),
        ("cell 43",  "Flight 1 scored plots shape", A1_clean.shape, (565, 35)),
        ("cell 43",  "Flight 1 diseased / moderate / healthy", class_counts(A1_clean), (25, 55, 485)),
        ("cell 221", "Flight 1 plots per raw score 1-10", score_counts(A1_clean),
         {1: 0, 2: 8, 3: 7, 4: 6, 5: 2, 6: 2, 7: 20, 8: 19, 9: 16, 10: 485}),
        ("cell 128", "texture table shape", texture_shape, (572, 7)),
        ("cell 130", "Flight 2 table with texture shape", (A2_tex.shape[0], A2_tex.shape[1] - 3), (527, 38)),
        ("cell 130", "Flight 1 table with texture shape", (A1_tex.shape[0], A1_tex.shape[1] - 3), (565, 38)),
        ("cell 130", "missing texture values, both flights",
         int(A2_tex[TEXTURE_COLS].isna().sum().sum() + A1_tex[TEXTURE_COLS].isna().sum().sum()), 0),
        ("cell 134", "texture z-scores have mean 0 and SD 1 in both flights",
         bool(all(np.allclose(d[norm_cols].mean(), 0, atol=1e-9) and np.allclose(d[norm_cols].std(), 1)
                  for d in (A2_tex, A1_tex))), True),
    ]

    print("SANITY CHECK against values printed in my_code.html")
    all_ok = True
    for cell, name, got, expected in checks:
        ok = got == expected
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {cell:8s} {name}")
        if not ok:
            print(f"             expected {expected}\n             got      {got}")

    if not all_ok and not args.force:
        print("\nAt least one check failed. Nothing written. Investigate before trusting any output.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Write, without overwriting an existing table that differs
    # -----------------------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    key_cols = ["Disease_Plot_ID", "Variety_ID", "Disease_class", "NDRE_mean", "canopy_cover_mean",
                "GNDVI_mean", "TCARI_OSAVI_mean"] + norm_cols
    for name, table in [("A2_clean_tex_norm.csv", A2_tex), ("A1_clean_tex_norm.csv", A1_tex)]:
        target = PROCESSED_DIR / name
        if target.exists():
            old = pd.read_csv(target)
            same = (all(c in old.columns for c in key_cols)
                    and old.shape[0] == table.shape[0]
                    and list(old["Disease_Plot_ID"].astype(str)) == list(table["Disease_Plot_ID"].astype(str))
                    and list(old["Variety_ID"].astype(str)) == list(table["Variety_ID"].astype(str))
                    and all(np.allclose(old[c], table[c], equal_nan=True) for c in key_cols[2:]))
            if not same:
                alt = target.with_name(target.stem + "_rebuilt.csv")
                table.to_csv(alt, index=False)
                print(f"\n{name}: DIFFERS from the existing file. Existing file kept; rebuilt copy at {alt}")
                continue
            print(f"\n{name}: identical to the existing file (row order, varieties, model columns).")
        table.to_csv(target, index=False)
        print(f"Wrote {target}")

    print("\nDone." if all_ok else "\nWritten with --force despite failed checks. Do not use for results.")


if __name__ == "__main__":
    main()
