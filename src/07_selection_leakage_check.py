"""
07_selection_leakage_check.py
=============================
Measures how much the locked model owes to having chosen its texture features
while looking at the test flight.

Run from the repo root, after 01_build_plot_table.py:
    python src/07_selection_leakage_check.py

The problem, in plain words
---------------------------
The three texture features in the locked model were picked by ranking twenty
candidates on how well each separated diseased from healthy plots. That ranking
used the disease labels of BOTH flights -- including Flight 1, which is the
held-out external test flight. So the features were chosen with the answer
sheet open. The external macro-F1 of 0.394 is therefore not quite the clean
out-of-sample number it appears to be. See preprocessing/README.md, step 16.

The effect is probably small: three of seven features, picked by a coarse
criterion, not fitted coefficients. But "probably small" is not a measurement,
so this script measures it.

What it does
------------
1. Redoes the selection using Flight 2 labels ONLY, the flight the model is
   allowed to train on, and prints the whole ranking rather than just the
   winners, so the margins are visible.
2. Says whether the leak-free selection picks the same three features.
3. Refits the locked two-stage cascade with the leak-free texture features
   substituted for the original three. Everything else is identical: same four
   spectral features, same SVM, same folds, same seed.
4. Refits again with the four low-VIF spectral features and no texture at all,
   which brackets the question from the other side: if dropping texture
   entirely costs little, then how texture was chosen matters little either.

A note on what "leak-free" can mean with one flight
---------------------------------------------------
The original criterion asks whether a feature's diseased-minus-healthy
difference points the same way in both flights. With only Flight 2 available
there is no second flight to agree with, so that test cannot be run at all.
What survives is the effect size itself, on Flight 2, and the two unwritten
rules the original selection also used: one statistic per band, Green excluded,
and each statistic used at most once. Those are applied here unchanged, so the
only thing that differs from the original is which labels were allowed to be
seen.

There are NO pass/fail checks in this script, and no expected values, because
no authorised value exists for any number it prints. Nothing here has been
traced to the thesis. It is a diagnostic, and it always exits 0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

TEXTURE_FULL_FILE = "texture_features_percentiles.csv"

LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
ORIGINAL_TEXTURE = ["NIR_max", "RedEdge_p90", "Red_min"]      # chosen using both flights
BANDS = ["Green", "Red", "RedEdge", "NIR"]
BAND_STATS = ["p10", "p90", "std", "min", "max"]
EXCLUDED_BAND = "Green"          # excluded by the original selection, kept excluded here

LO = ["diseased", "moderate", "healthy"]
ORD = {"diseased": 0, "moderate": 1, "healthy": 2}

# What the locked model scores, for side-by-side printing only. Not checks.
LOCKED = {"within": (0.438, 0.206, 53, 86), "external": (0.394, 0.206, 8, 25)}


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def prep(Xtr, Xte):
    im = SimpleImputer(strategy="median")
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def svc():
    return SVC(kernel="linear", class_weight="balanced", C=0.1)


def cascade(Xtr, ytr, Xte):
    a = svc().fit(Xtr, (ytr == "healthy").astype(int)).predict(Xte)
    b = svc().fit(Xtr, (ytr == "diseased").astype(int)).predict(Xte)
    return np.array(["healthy" if i == 1 else ("diseased" if j == 1 else "moderate")
                     for i, j in zip(a, b)])


def evaluate(A2, A1, features):
    """Within-flight grouped CV and external transfer, for one feature set."""
    y2, y1 = A2["y3"].values, A1["y3"].values
    X2, X1 = A2[features].values, A1[features].values
    g2 = A2["Variety_ID"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    T, P = [], []
    for tr, te in cv.split(X2, y2, g2):
        a, b = prep(X2[tr], X2[te])
        P += list(cascade(a, y2[tr], b))
        T += list(y2[te])
    T, P = np.array(T), np.array(P)

    a, b = prep(X2, X1)
    Pe = cascade(a, y2, b)

    out = {}
    for name, truth, pred in (("within", T, P), ("external", y1, Pe)):
        dm = truth == "diseased"
        out[name] = (
            f1_score(truth, pred, average="macro", labels=LO),
            cohen_kappa_score([ORD[t] for t in truth], [ORD[p] for p in pred],
                              weights="quadratic"),
            int((pred[dm] == "diseased").sum()),
            int(dm.sum()),
        )
    return out


def main():
    A2 = pd.read_csv(require(PROCESSED_DIR / "A2_clean_tex_norm.csv"))
    A1 = pd.read_csv(require(PROCESSED_DIR / "A1_clean_tex_norm.csv"))
    for d in (A2, A1):
        d["y3"] = d["Disease_class"].apply(lab)
        d["Disease_Plot_ID"] = d["Disease_Plot_ID"].astype(str)

    texture_path = DATA_DIR / TEXTURE_FULL_FILE
    if not texture_path.exists():
        print(f"Missing input: {texture_path}")
        print("The selection cannot be redone without the full texture table. Nothing to do.")
        sys.exit(0)
    texture = pd.read_csv(texture_path)
    texture["Disease_Plot_ID"] = texture["Disease_Plot_ID"].astype(str)

    print("=" * 74)
    print("1. REDOING THE TEXTURE SELECTION WITH FLIGHT 2 LABELS ONLY")
    print("=" * 74)

    # Flight 2 labels, joined to the raw (un-normalised) texture columns, exactly
    # as the original selection did.
    labels2 = A2[["Disease_Plot_ID", "y3"]]
    merged = texture.merge(labels2, on="Disease_Plot_ID", how="inner")
    print(f"   n = {len(merged)} scored plots from the training flight")
    print(f"   candidates = {len(BANDS)} bands x {len(BAND_STATS)} statistics = "
          f"{len(BANDS) * len(BAND_STATS)}\n")

    rows = []
    for band in BANDS:
        for stat in BAND_STATS:
            col = f"{band}_2_{stat}"
            groups = merged.groupby("y3")[col].mean()
            diff = groups["diseased"] - groups["healthy"]
            sd = texture[col].std()
            rows.append({"band": band, "stat": stat, "diff_flight2": diff,
                         "normalised_effect": abs(diff / sd) if sd else np.nan})
    ranking = pd.DataFrame(rows).sort_values("normalised_effect", ascending=False)

    print("   full ranking, Flight 2 only (diseased minus healthy, normalised by the column SD)")
    print(f"   {'band':9s} {'stat':5s} {'diff (Flight 2)':>17s} {'normalised effect':>19s}")
    for _, r in ranking.iterrows():
        mark = "  <- excluded band" if r["band"] == EXCLUDED_BAND else ""
        print(f"   {r['band']:9s} {r['stat']:5s} {r['diff_flight2']:17.6f}"
              f" {r['normalised_effect']:19.6f}{mark}")

    # The original's two unwritten rules, applied unchanged: one statistic per
    # band, Green excluded, and no statistic reused across bands.
    chosen, used_bands, used_stats = [], set(), set()
    for _, r in ranking.iterrows():
        if r["band"] == EXCLUDED_BAND or r["band"] in used_bands or r["stat"] in used_stats:
            continue
        chosen.append(f"{r['band']}_{r['stat']}")
        used_bands.add(r["band"])
        used_stats.add(r["stat"])
        if len(chosen) == len(ORIGINAL_TEXTURE):
            break

    print(f"\n   leak-free selection : {', '.join(chosen)}")
    print(f"   original selection  : {', '.join(ORIGINAL_TEXTURE)}")
    same = set(chosen) == set(ORIGINAL_TEXTURE)
    print(f"   same three features : {'YES' if same else 'NO'}")
    if not same:
        gained = sorted(set(chosen) - set(ORIGINAL_TEXTURE))
        lost = sorted(set(ORIGINAL_TEXTURE) - set(chosen))
        print(f"     dropped by the leak-free rule : {', '.join(lost)}")
        print(f"     picked instead                : {', '.join(gained)}")

    # -- z-score the leak-free features within each flight, as 01 does ---------
    def attach(df, flight):
        out = df.copy()
        for feat in chosen:
            band, stat = feat.rsplit("_", 1)
            col = f"{band}_{flight}_{stat}"
            vals = texture[["Disease_Plot_ID", col]].rename(columns={col: feat + "_lf"})
            out = out.merge(vals, on="Disease_Plot_ID", how="left")
            v = out[feat + "_lf"]
            out[feat + "_lf_norm"] = (v - v.mean()) / v.std()
        return out

    A2f, A1f = attach(A2, "2"), attach(A1, "1")
    leakfree_features = LOW_VIF + [f + "_lf_norm" for f in chosen]
    original_features = LOW_VIF + [f + "_norm" for f in ORIGINAL_TEXTURE]
    spectral_only = LOW_VIF

    print("\n" + "=" * 74)
    print("2. THE LOCKED CASCADE REFITTED, THREE FEATURE SETS")
    print("=" * 74)
    print("   everything except the feature list is identical: same SVM, C=0.1,")
    print("   class_weight='balanced', same 5 grouped folds, same seed\n")

    variants = [
        ("locked, as published (7 features)", original_features, A2, A1),
        ("leak-free texture (7 features)", leakfree_features, A2f, A1f),
        ("no texture at all (4 spectral)", spectral_only, A2, A1),
    ]
    results = {}
    for name, feats, d2, d1 in variants:
        results[name] = evaluate(d2, d1, feats)

    header = f"   {'feature set':34s} {'macro-F1':>9s} {'QWK':>7s} {'diseased recall':>17s}"
    for cond in ("within", "external"):
        print(f"   {cond.upper()}-FLIGHT" if cond == "within" else f"\n   EXTERNAL (Flight 2 -> Flight 1)")
        print(header)
        lf1, lqwk, lrec, ltot = LOCKED[cond]
        print(f"   {'locked model, thesis value':34s} {lf1:9.3f} {lqwk:7.3f}"
              f" {str(lrec) + '/' + str(ltot):>17s}")
        for name, _, _, _ in variants:
            f1v, qwkv, rec, tot = results[name][cond]
            print(f"   {name:34s} {f1v:9.3f} {qwkv:7.3f} {str(rec) + '/' + str(tot):>17s}")

    # -- plain-language summary ----------------------------------------------
    print("\n" + "=" * 74)
    print("3. WHAT CHANGED AND WHAT DID NOT")
    print("=" * 74)
    lk = results["locked, as published (7 features)"]
    lf = results["leak-free texture (7 features)"]
    sp = results["no texture at all (4 spectral)"]

    def delta(a, b, i=0):
        return b[i] - a[i]

    print(f"   Selecting the texture features without the test flight "
          f"{'changed which three were picked' if not same else 'picked the same three'}.")
    if same:
        print("   With the same three features, nothing downstream can change: the leak-free")
        print("   and published models are the same model. The selection step did not depend")
        print("   on the test-flight labels after all.")
    else:
        print(f"   Within-flight macro-F1 moved {delta(lk['within'], lf['within']):+.3f}"
              f" (from {lk['within'][0]:.3f} to {lf['within'][0]:.3f}).")
        print(f"   External macro-F1 moved {delta(lk['external'], lf['external']):+.3f}"
              f" (from {lk['external'][0]:.3f} to {lf['external'][0]:.3f}).")
        print(f"   External diseased recall went from {lk['external'][2]}/{lk['external'][3]}"
              f" to {lf['external'][2]}/{lf['external'][3]}.")
        if lf["external"][0] < lk["external"][0]:
            print("   The external score is LOWER without the leak, which is the direction that")
            print("   matters: part of the published external number came from having chosen")
            print("   the features with the test flight's labels visible.")
        else:
            print("   The external score is NO LOWER without the leak, so the published number")
            print("   does not appear to depend on having seen the test-flight labels.")
    print()
    print(f"   Dropping texture entirely costs {lk['external'][0] - sp['external'][0]:+.3f}"
          f" external macro-F1 ({lk['external'][0]:.3f} -> {sp['external'][0]:.3f})"
          f" and takes diseased recall from {lk['external'][2]}/{lk['external'][3]}"
          f" to {sp['external'][2]}/{sp['external'][3]}.")
    print("   That is the size of the prize the whole texture pipeline is competing for, and")
    print("   it bounds how much the selection method could matter either way.")
    print()
    print("   None of these numbers is checked against anything. No authorised value exists")
    print("   for any of them. Treat them as a diagnostic, not as results.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"feature_set": name, "condition": cond,
         "macro_f1": results[name][cond][0], "qwk": results[name][cond][1],
         "diseased_recalled": results[name][cond][2], "diseased_total": results[name][cond][3]}
        for name, _, _, _ in variants for cond in ("within", "external")
    ]).to_csv(OUT_DIR / "selection_leakage_check.csv", index=False)
    ranking.to_csv(OUT_DIR / "texture_selection_flight2_only.csv", index=False)
    sys.exit(0)


if __name__ == "__main__":
    main()
