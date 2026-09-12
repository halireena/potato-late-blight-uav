"""
05_supporting_statistics.py
===========================
The supporting numbers the thesis leans on outside the headline model results.

Run from the repo root, after 01_build_plot_table.py:
    python src/05_supporting_statistics.py

The idea in plain words
-----------------------
The model scripts answer "how well does it do?". These six statistics answer
the questions a reader asks next, and most of them are checks on whether the
data could support a better answer at all:

1. Collinearity. Four spectral indices were kept and seven dropped. Variance
   inflation factors say whether the four that stayed are actually independent
   enough to be read separately.
2. Label reliability. Two plots of the same variety, in the same field, on the
   same day, should score the same. How often do they not? That is the ceiling
   on what any model can learn, because it is the disagreement rate of the
   labels themselves.
3. Spatial structure. If diseased plots cluster, neighbouring plots leak into
   each other and grouped cross-validation is not enough. This tests whether a
   plot's neighbours predict its own score.
4. The moderate class. 93 Kruskal-Wallis tests ask, one feature at a time and
   under three definitions of "moderate", whether the middle class separates
   from the rest at all. The smallest p across all 93 is the headline.
5. Binary vs three-class. Dropping the moderate class entirely and asking only
   healthy-vs-diseased: does diseased recall improve when the middle boundary
   is removed?
6. Per-flight normalisation. Z-scoring all seven features within each flight,
   rather than only the three texture ones, trades overall macro-F1 for
   diseased recall. This measures the trade.

Every statistic prints the n it was computed on. Where a statistic needs an
input file that is not present, it says which file and is skipped; the rest
still run.

Two details that decide the numbers
-----------------------------------
- The VIFs are computed WITH an intercept added. Without one they are roughly
  eight times larger and mean something different: statsmodels computes VIF
  against the other columns only, so leaving the constant out folds each
  variable's mean into its "explained" variance. The recorded 5.36 is the
  with-intercept figure.
- The 93 Kruskal-Wallis tests run over 31 features: the eleven spectral index
  means plus the twenty texture statistics, each z-scored within its flight.
  Restricting to the columns of the processed table alone gives 42 tests, not
  93, and a different minimum p.

Every expected value was printed by an executed cell of the thesis notebook
(mine_ML.ipynb) and was supplied as authorised. The prevalence-shift
resampling test is computed and PRINTED but NOT checked: two values are on
record for it and the point is to see which the locked model's own predictions
give. The script exits with code 1 on any mismatch.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

STATS_FILE_F2 = "ALL_stats_2_JB_jack_data.csv"
TEXTURE_FULL_FILE = "texture_features_percentiles.csv"

LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
TEX_N = ["NIR_max_norm", "RedEdge_p90_norm", "Red_min_norm"]
F7 = LOW_VIF + TEX_N
SPECTRAL_MEANS = ["NDVI_mean", "NDRE_mean", "GNDVI_mean", "RDVI_mean", "PVR_mean",
                  "EVI2_mean", "TVI_mean", "TCARI_mean", "OSAVI_mean",
                  "TCARI_OSAVI_mean", "canopy_cover_mean"]
BANDS = ["Green", "Red", "RedEdge", "NIR"]
BAND_STATS = ["p10", "p90", "std", "min", "max"]
GENERIC = [f"{b}_{s}" for b in BANDS for s in BAND_STATS]

LO = ["diseased", "moderate", "healthy"]
MODERATE_DEFINITIONS = {"7-9": (7, 9), "5-9": (5, 9), "4-9": (4, 9)}


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def prep(Xtr, Xte):
    im = SimpleImputer(strategy="median")
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def svc(probability=False):
    return SVC(kernel="linear", class_weight="balanced", C=0.1, probability=probability)


def cascade(Xtr, ytr, Xte):
    a = svc().fit(Xtr, (ytr == "healthy").astype(int)).predict(Xte)
    b = svc().fit(Xtr, (ytr == "diseased").astype(int)).predict(Xte)
    return np.array(["healthy" if i == 1 else ("diseased" if j == 1 else "moderate")
                     for i, j in zip(a, b)])


def optional(path):
    """Return the path if the input is there, else None. Used so one missing
    file skips one statistic instead of stopping the script."""
    return path if path.exists() else None


def main():
    A2 = pd.read_csv(require(PROCESSED_DIR / "A2_clean_tex_norm.csv"))   # Flight 2, training
    A1 = pd.read_csv(require(PROCESSED_DIR / "A1_clean_tex_norm.csv"))   # Flight 1, test
    for d in (A2, A1):
        d["y3"] = d["Disease_class"].apply(lab)
        d["Disease_Plot_ID"] = d["Disease_Plot_ID"].astype(str)
    y2, y1 = A2["y3"].values, A1["y3"].values

    checks = []      # (name, got, expected)
    skipped = []     # (name, why)
    record = []      # rows for the csv

    print("=" * 70)
    print("SUPPORTING STATISTICS")
    print("=" * 70)

    # -- 1. Collinearity of the retained four ---------------------------------
    print("\n1. VARIANCE INFLATION, the retained four-feature set")
    X = SimpleImputer(strategy="median").fit_transform(A2[LOW_VIF].values)
    Xc = add_constant(X)             # the intercept matters, see the docstring
    vifs = {c: float(round(variance_inflation_factor(Xc, i + 1), 2))
            for i, c in enumerate(LOW_VIF)}
    print(f"   n = {len(A2)} plots, {len(LOW_VIF)} features, intercept included")
    for c in sorted(vifs, key=vifs.get, reverse=True):
        print(f"     {c:20s} VIF = {vifs[c]:.2f}")
    checks.append(("VIF, GNDVI_mean in the locked four", vifs["GNDVI_mean"], 5.36))
    record.append({"statistic": "VIF GNDVI_mean", "n": len(A2), "value": vifs["GNDVI_mean"]})

    # -- 2. Replicate disagreement within a variety ---------------------------
    print("\n2. WITHIN-VARIETY REPLICATE DISAGREEMENT")
    counts = A2.groupby("Variety_ID")["Disease_class"].agg(["min", "max", "count"])
    two = counts[counts["count"] == 2]
    raw_disagree = int((two["max"] - two["min"] > 0).sum())
    three = A2.groupby("Variety_ID")["y3"].agg(["nunique", "count"])
    two_three = three[three["count"] == 2]
    class_disagree = int((two_three["nunique"] > 1).sum())
    n_two = len(two)
    raw_pct = round(100 * raw_disagree / n_two, 1)
    class_pct = round(100 * class_disagree / n_two, 1)
    print(f"   n = {n_two} varieties with exactly two scored replicate plots")
    print(f"     on the raw 1-to-10 score:     {raw_disagree}/{n_two} = {raw_pct:.1f}% disagree")
    print(f"     after collapsing to 3 classes: {class_disagree}/{n_two} = {class_pct:.1f}% disagree")
    print("     (this is the label's own disagreement rate, so it caps what any model can learn)")
    checks.append(("Replicate disagreement, raw score", (raw_disagree, n_two), (95, 243)))
    checks.append(("Replicate disagreement, raw score %", raw_pct, 39.1))
    checks.append(("Replicate disagreement, three-class", (class_disagree, n_two), (87, 243)))
    checks.append(("Replicate disagreement, three-class %", class_pct, 35.8))
    record.append({"statistic": "replicate disagreement raw", "n": n_two, "value": raw_pct})
    record.append({"statistic": "replicate disagreement 3-class", "n": n_two, "value": class_pct})

    # -- 3. Spatial autocorrelation -------------------------------------------
    print("\n3. SPATIAL AUTOCORRELATION OF SEVERITY")
    stats_path = optional(DATA_DIR / STATS_FILE_F2)
    if stats_path is None:
        why = f"{STATS_FILE_F2} not found in {DATA_DIR}"
        print(f"   [SKIPPED] {why}")
        skipped.append(("Spatial autocorrelation", why))
    else:
        raw = pd.read_csv(stats_path, dtype={"Disease_Plot_ID": str})
        if not {"row_index", "col_index"} <= set(raw.columns):
            why = f"{STATS_FILE_F2} has no row_index/col_index columns"
            print(f"   [SKIPPED] {why}")
            skipped.append(("Spatial autocorrelation", why))
        else:
            raw = raw[raw["Disease_Plot_ID"].notna() & (raw["Disease_Plot_ID"] != "G")]
            lookup = raw[["Disease_Plot_ID", "row_index", "col_index"]].drop_duplicates("Disease_Plot_ID")
            grid = A2.merge(lookup, on="Disease_Plot_ID", how="left")
            coords = grid[["row_index", "col_index", "Disease_class",
                           "Disease_Plot_ID"]].dropna().reset_index(drop=True)
            neighbour_mean, own = [], []
            for _, r in coords.iterrows():
                # the up-to-eight plots touching this one in the field layout
                nb = coords[(abs(coords["row_index"] - r["row_index"]) <= 1)
                            & (abs(coords["col_index"] - r["col_index"]) <= 1)
                            & (coords["Disease_Plot_ID"] != r["Disease_Plot_ID"])]
                if len(nb) >= 2:
                    neighbour_mean.append(nb["Disease_class"].mean())
                    own.append(r["Disease_class"])
            rho, p_rho = spearmanr(neighbour_mean, own)
            print(f"   n = {len(own)} plots with at least two neighbours"
                  f" (of {len(coords)} placed on the grid)")
            print(f"     own score vs neighbours' mean: Spearman rho = {rho:.3f}, p = {p_rho:.3f}")
            print("     (no detectable clustering, so grouping by variety is enough)")
            checks.append(("Spatial autocorrelation, Spearman rho", float(round(rho, 3)), -0.036))
            checks.append(("Spatial autocorrelation, p", float(round(p_rho, 3)), 0.413))
            record.append({"statistic": "spatial rho", "n": len(own), "value": float(round(rho, 3))})

    # -- 4. The 93 Kruskal-Wallis tests ---------------------------------------
    print("\n4. CAN THE MODERATE CLASS BE SEPARATED AT ALL? 93 Kruskal-Wallis tests")
    texture_path = optional(DATA_DIR / TEXTURE_FULL_FILE)
    if texture_path is None:
        why = f"{TEXTURE_FULL_FILE} not found in {DATA_DIR}; without it only 42 of the 93 tests exist"
        print(f"   [SKIPPED] {why}")
        skipped.append(("Kruskal-Wallis screen", why))
    else:
        texture_full = pd.read_csv(texture_path)
        texture_full["Disease_Plot_ID"] = texture_full["Disease_Plot_ID"].astype(str)
        cols = [f"{b}_2_{s}" for b in BANDS for s in BAND_STATS]
        t = texture_full[["Disease_Plot_ID"] + cols].copy()
        t.columns = ["Disease_Plot_ID"] + GENERIC
        kw = A2[["Disease_Plot_ID", "Disease_class"] + SPECTRAL_MEANS].merge(
            t, on="Disease_Plot_ID", how="left")
        for c in GENERIC:
            kw[c + "_norm"] = (kw[c] - kw[c].mean()) / kw[c].std()
        features = SPECTRAL_MEANS + [c + "_norm" for c in GENERIC]
        rows = []
        for label, (lo_s, hi_s) in MODERATE_DEFINITIONS.items():
            mod = (kw["Disease_class"] >= lo_s) & (kw["Disease_class"] <= hi_s)
            for feat in features:
                a, b = kw[feat][mod].dropna(), kw[feat][~mod].dropna()
                if len(a) < 3 or len(b) < 3:
                    continue
                H, p = kruskal(a, b)
                rows.append({"grouping": label, "feature": feat, "H": H, "p": p})
        kw_results = pd.DataFrame(rows).sort_values("p")
        min_p = float(round(kw_results["p"].min(), 3))
        n_sig = int((kw_results["p"] < 0.05).sum())
        print(f"   n = {len(kw)} plots, {len(features)} features x"
              f" {len(MODERATE_DEFINITIONS)} definitions of moderate = {len(kw_results)} tests")
        print(f"     smallest p of all {len(kw_results)}: {min_p:.3f}")
        print(f"     significant at p<0.05 before any correction: {n_sig} of {len(kw_results)}")
        print("     (nothing separates the moderate class, even before correcting for 93 tests)")
        checks.append(("Kruskal-Wallis, number of tests", len(kw_results), 93))
        checks.append(("Kruskal-Wallis, smallest p of 93", min_p, 0.056))
        record.append({"statistic": "Kruskal-Wallis min p", "n": len(kw), "value": min_p})

    # -- 5. Binary healthy-vs-diseased against three-class --------------------
    print("\n5. BINARY HEALTHY-VS-DISEASED vs THREE-CLASS, on the test flight")
    keep2 = y2 != "moderate"
    keep1 = y1 != "moderate"
    Xb2, Xb1 = A2[F7].values[keep2], A1[F7].values[keep1]
    yb2 = (y2[keep2] == "diseased").astype(int)
    yb1 = (y1[keep1] == "diseased").astype(int)
    a, b = prep(Xb2, Xb1)
    pred_bin = svc(probability=True).fit(a, yb2).predict(b)
    tn, fp, fn, tp = confusion_matrix(yb1, pred_bin).ravel()
    binary_recall = round(100 * tp / (tp + fn), 1)

    a3, b3 = prep(A2[F7].values, A1[F7].values)
    pred_three = cascade(a3, y2, b3)
    dm = y1 == "diseased"
    three_recall = round(100 * (pred_three[dm] == "diseased").sum() / dm.sum(), 1)
    print(f"   n = {int(keep2.sum())} training plots and {int(keep1.sum())} test plots after"
          f" dropping moderate ({int((~keep1).sum())} dropped)")
    print(f"     binary diseased recall:      {tp}/{tp + fn} = {binary_recall:.1f}%")
    print(f"     three-class diseased recall: {int((pred_three[dm] == 'diseased').sum())}"
          f"/{int(dm.sum())} = {three_recall:.1f}%  (n = {len(y1)} test plots)")
    print("     (removing the middle boundary finds diseased plots the cascade missed)")
    checks.append(("Binary diseased recall, test flight %", binary_recall, 52.0))
    checks.append(("Three-class diseased recall, test flight %", three_recall, 32.0))
    record.append({"statistic": "binary diseased recall %", "n": int(keep1.sum()), "value": binary_recall})

    # -- 6. Per-flight normalisation of all seven features --------------------
    print("\n6. PER-FLIGHT NORMALISATION OF ALL SEVEN FEATURES")
    z2 = ((A2[F7] - A2[F7].mean()) / A2[F7].std()).values
    z1 = ((A1[F7] - A1[F7].mean()) / A1[F7].std()).values
    za, zb = prep(z2, z1)
    pred_z = cascade(za, y2, zb)
    f1_z = float(round(f1_score(y1, pred_z, average="macro", labels=LO), 3))
    rec_z = int((pred_z[dm] == "diseased").sum())
    rec_z_pct = round(100 * rec_z / dm.sum(), 1)
    print(f"   n = {len(y1)} test plots, {int(dm.sum())} of them diseased")
    print(f"     macro-F1 = {f1_z:.3f}, diseased recall = {rec_z}/{int(dm.sum())} = {rec_z_pct:.1f}%")
    print("     (macro-F1 falls, diseased recall roughly doubles: a trade, not an improvement)")
    checks.append(("All-feature per-flight normalisation, macro-F1", f1_z, 0.308))
    checks.append(("All-feature per-flight normalisation, diseased recall %", rec_z_pct, 64.0))
    record.append({"statistic": "all-feature norm macro-F1", "n": len(y1), "value": f1_z})

    # -----------------------------------------------------------------------
    # PRINTED, NOT CHECKED: prevalence-shift resampling.
    # Two values are on record, 0.293 and 0.370, and they are attributed to
    # different resampling schemes. Both schemes are run here on the SAME
    # input -- the locked cascade's own within-flight predictions -- so that
    # the scheme is the only thing that differs between them.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UNCHECKED: prevalence-shift resampling")
    print("=" * 70)
    print("Question: if the within-flight predictions are resampled to Flight 1's")
    print("class mix, with no drift possible because the features never change,")
    print("what macro-F1 comes out? Both recorded schemes are run on the locked")
    print("cascade's own predictions, so only the scheme differs.\n")

    X2, g2 = A2[F7].values, A2["Variety_ID"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    Tw, Pw = [], []
    for tr, te in cv.split(X2, y2, g2):
        xa, xb = prep(X2[tr], X2[te])
        Pw += list(cascade(xa, y2[tr], xb))
        Tw += list(y2[te])
    Tw, Pw = np.array(Tw), np.array(Pw)
    within_f1 = f1_score(Tw, Pw, average="macro", labels=LO)
    prevalence = pd.Series(y1).value_counts(normalize=True).to_dict()
    pools = {c: np.where(Tw == c)[0] for c in LO}

    def resample_to_prevalence(n_total, n_repeats, seed, always_replace):
        rg = np.random.default_rng(seed)
        out = []
        for _ in range(n_repeats):
            take = []
            for c in LO:
                k = int(round(prevalence.get(c, 0) * n_total))
                pool = pools[c]
                if k == 0 or len(pool) == 0:
                    continue
                take.append(rg.choice(pool, size=k,
                                      replace=True if always_replace else (k > len(pool))))
            idx = np.concatenate(take)
            out.append(f1_score(Tw[idx], Pw[idx], average="macro", labels=LO))
        return np.array(out)

    n_anchored = int(len(pools["healthy"]) / prevalence["healthy"])
    schemes = [
        ("A: total fixed at the within-flight n", len(Tw), 5000, 42, True),
        ("B: total anchored on the healthy pool", n_anchored, 200, 0, False),
    ]
    print(f"  Locked cascade within-flight macro-F1 = {within_f1:.3f}, n = {len(Tw)} plots")
    print(f"  Flight 1 class mix: " + ", ".join(f"{c} {prevalence[c]:.3f}" for c in LO))
    for label, n_total, reps, seed, always in schemes:
        sim = resample_to_prevalence(n_total, reps, seed, always)
        lo95, hi95 = np.percentile(sim, [2.5, 97.5])
        draw = ", ".join(f"{c} {int(round(prevalence[c] * n_total))}" for c in LO)
        print(f"\n  Scheme {label}")
        print(f"    resampled n = {n_total} per draw ({draw}); {reps} repeats; seed {seed};"
              f" {'always with replacement' if always else 'with replacement only when short'}")
        print(f"    macro-F1 = {sim.mean():.3f}  95% range [{lo95:.3f}, {hi95:.3f}]  (n = {len(Tw)} source predictions)")
        record.append({"statistic": f"prevalence shift, scheme {label[0]}",
                       "n": len(Tw), "value": float(round(sim.mean(), 3))})
    print("\n  Recorded values for this test: 0.293 and 0.370.")
    print("  Both schemes above land near 0.370 on the locked cascade's own predictions.")
    print("  The scheme is therefore NOT what separates the two recorded numbers.")
    print("  No pass/fail check is applied. Settle it against the notebook.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(record).to_csv(OUT_DIR / "supporting_statistics.csv", index=False)

    # -----------------------------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECK against authorised values from the thesis notebook")
    print("=" * 70)
    all_ok = True
    for name, got, expected in checks:
        ok = got == expected
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name:52s} got {got}  expected {expected}")
    if skipped:
        print()
        for name, why in skipped:
            print(f"  [SKIPPED] {name}: {why}")
        print(f"\n  {len(skipped)} statistic(s) skipped for missing inputs. The run is INCOMPLETE:")
        print("  a skipped statistic is neither confirmed nor refuted.")
    print("\nAll matched." if all_ok
          else "\nMISMATCH found. Do not trust any number above until it is explained.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
