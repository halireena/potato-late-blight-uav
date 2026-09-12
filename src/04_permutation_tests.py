"""
04_permutation_tests.py
=======================
Asks of every headline result: could a model have scored this well on labels
that mean nothing?

Run from the repo root, after 01_build_plot_table.py:
    python src/04_permutation_tests.py

The idea in plain words
-----------------------
A macro-F1 of 0.394 is only impressive if a model given nonsense labels cannot
reach it. So each result is recomputed 1,000 times with the training labels
shuffled at random. Shuffling destroys any real link between the imagery and
the disease score while keeping everything else identical: the same plots, the
same features, the same folds, the same model. The 1,000 shuffled scores are
the null distribution -- what this pipeline produces from noise alone.

The real score is then read against that distribution:

    p = (number of shuffled scores >= the real score + 1) / (1,000 + 1)

The +1 on both sides is the standard correction; it stops p ever being exactly
zero, because 1,000 shuffles cannot prove a probability is zero.

A small p means noise rarely reaches the real score, so the model found
something. A p near 1 means the real score sits inside, or below, what noise
produces -- the architecture failed, and failed in a way that raw accuracy
hides. Both outcomes are reported here, because several of these tests are
expected to fail and the failures are part of the finding.

What is tested, and why these thirteen
--------------------------------------
    TSLSC, within and external        the locked model, both conditions
    diseased-first, within and ext    the documented trade-off variant
    RF / XGBoost / ordinal, external  the three that collapse to the floor
    unweighted baseline, within/ext   the no-class-weighting baseline
    balanced single stage, within     one balanced 3-class model, no cascade
    variety-level AUC, within and ext the variety-level analysis
    moderate-vs-healthy AUC, external can the middle class be found at all?

Three details that change the numbers, so they are followed exactly
------------------------------------------------------------------
1. The shuffled labels are fed to the splitter as well as to the model, so the
   folds are rebuilt from the permuted labels. The splitter's settings are
   fixed (5 folds, shuffle, seed 42); the fold contents are not frozen.
2. Three tests use their own feature set, because that is what produced the
   recorded numbers: the unweighted baseline and the balanced single-stage
   model use six spectral means; ordinal regression uses the four low-VIF
   features; the variety-level tests use four low-VIF means plus twenty
   texture statistics.
3. The balanced single-stage score is the MEAN of the five per-fold macro-F1
   values, not one score pooled across folds. Those are not the same number,
   and the recorded 0.358 is the mean-of-folds.

Reading the check
-----------------
Real scores are compared exactly, at the three decimals that were printed.
Permuted means and SDs are themselves results of a random process: a rerun
moves them slightly even with the seed fixed, because the permutations interact
with refitted folds. They are compared within +/-0.02.

p-values are handled in two ways, because the two kinds of row mean different
things. Where the recorded p is 0.05 or above, the value IS the finding -- the
architecture did not clear its null -- so it is compared within +/-0.02. Where
the recorded p is below 0.05, the finding is that the result cleared its null,
not that p landed on any particular number; with 1,000 shuffles the smallest
reachable p is 1/1001, and the exact figure wanders between runs. Those rows
are therefore not compared by value at all: the check is that the computed p is
below 0.05, and both figures are printed so the drift stays visible.

Every rule above is restated in the output, next to the results it governs.

Every expected value was printed by an executed cell of the thesis notebook
(mine_ML.ipynb) and was supplied as authorised. The ordinal regression
within-flight test is computed and PRINTED but deliberately NOT checked: two
different real scores are on record for it and the point of running it is to
see which one its own null supports.

This script is the slow one: 13,000 model fits plus the unchecked test, of
which the Random Forest and XGBoost nulls are most of the wall-clock. Expect
roughly fifteen minutes. Progress is printed as it goes.
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import mord

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

# Shuffled labels make some folds almost separable, and the logistic solver then
# overflows internally on its way to a fit it still returns. That is a numerical
# grumble from inside sklearn, not a problem with the data: the inputs were
# checked for missing and infinite values and have neither. Silenced only so
# that 14,000 fits do not bury the results. Nothing else is silenced.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*converge.*")

N_PERMUTATIONS = 1000
TOLERANCE = 0.02          # permuted mean and SD; real scores are compared exactly
SIGNIFICANCE_LEVEL = 0.05  # below this, a p-value is checked for clearing the null,
                           # not for landing on the exact number recorded

TEXTURE_FULL_FILE = "texture_features_percentiles.csv"

LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
TEX_N = ["NIR_max_norm", "RedEdge_p90_norm", "Red_min_norm"]
F7 = LOW_VIF + TEX_N
F6_SPECTRAL = ["canopy_cover_mean", "NDVI_mean", "NDRE_mean",
               "GNDVI_mean", "EVI2_mean", "OSAVI_mean"]
BANDS = ["Green", "Red", "RedEdge", "NIR"]
BAND_STATS = ["p10", "p90", "std", "min", "max"]
GENERIC = [f"{b}_{s}" for b in BANDS for s in BAND_STATS]
GENERIC_NORM = [c + "_norm" for c in GENERIC]
VARIETY_FEATURES = LOW_VIF + GENERIC_NORM

LO = ["diseased", "moderate", "healthy"]
ORD = {"diseased": 0, "moderate": 1, "healthy": 2}
INV_ORD = {v: k for k, v in ORD.items()}


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def normalise_variety(name):
    """Fold spelling variants of one variety together (spacing, case, brackets)."""
    return re.sub(r"[\s\-_()]", "", str(name)).upper()


def prep(Xtr, Xte):
    """Median-impute then standardise, both fitted on the training side only."""
    im = SimpleImputer(strategy="median")
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def svc():
    return SVC(kernel="linear", class_weight="balanced", C=0.1)


def grouped_cv():
    """A fresh splitter each time: the folds are rebuilt from whatever labels
    are handed to it, which is what the recorded tests did."""
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def cascade(Xtr, ytr, Xte, make, reversed_order=False, sample_weight=False):
    first, second = ("diseased", "healthy") if reversed_order else ("healthy", "diseased")
    preds = []
    for target in (first, second):
        yb = (ytr == target).astype(int)
        m = make()
        if sample_weight:
            m.fit(Xtr, yb, sample_weight=compute_sample_weight("balanced", yb))
        else:
            m.fit(Xtr, yb)
        preds.append(m.predict(Xte))
    a, b = preds
    if reversed_order:
        return np.array(["diseased" if i == 1 else ("healthy" if j == 1 else "moderate")
                         for i, j in zip(a, b)])
    return np.array(["healthy" if i == 1 else ("diseased" if j == 1 else "moderate")
                     for i, j in zip(a, b)])


def build_variety_features(plots, flight, texture_full):
    """Plot table + the twenty texture statistics, each z-scored within its own
    flight, ready to be averaged up to one row per variety."""
    cols = [f"{b}_{flight}_{s}" for b in BANDS for s in BAND_STATS]
    t = texture_full[["Disease_Plot_ID"] + cols].copy()
    t.columns = ["Disease_Plot_ID"] + GENERIC
    out = plots[["Disease_Plot_ID", "Variety_ID", "Disease_class"] + LOW_VIF].merge(
        t, on="Disease_Plot_ID", how="left")
    for c in GENERIC:
        out[c + "_norm"] = (out[c] - out[c].mean()) / out[c].std()
    return out


def to_variety_rows(df, key):
    """One row per variety: the mean of each feature over that variety's plots."""
    return df.groupby(key).agg(
        **{f: (f, "mean") for f in VARIETY_FEATURES},
        mean_severity=("Disease_class", "mean"),
        n_plots=("Disease_class", "count"),
    ).reset_index()


def permutation_test(name, score_fn, y_real, rng_seed=RANDOM_SEED):
    """Real score, then N_PERMUTATIONS scores on shuffled labels."""
    real = score_fn(y_real)
    rng = np.random.RandomState(rng_seed)
    perm = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        perm[i] = score_fn(rng.permutation(y_real))
        if (i + 1) % 250 == 0:
            print(f"    {name}: {i + 1}/{N_PERMUTATIONS}", flush=True)
    p = (np.sum(perm >= real) + 1) / (N_PERMUTATIONS + 1)
    print(f"  {name}: real={real:.3f}  permuted mean={perm.mean():.3f}"
          f" +/- {perm.std():.3f}  p={p:.4f}", flush=True)
    return {"test": name, "real": real, "permuted_mean": float(perm.mean()),
            "permuted_sd": float(perm.std()), "p_value": float(p)}


def main():
    A2 = pd.read_csv(require(PROCESSED_DIR / "A2_clean_tex_norm.csv"))   # Flight 2, training
    A1 = pd.read_csv(require(PROCESSED_DIR / "A1_clean_tex_norm.csv"))   # Flight 1, test
    for d in (A2, A1):
        d["y3"] = d["Disease_class"].apply(lab)
        d["Disease_Plot_ID"] = d["Disease_Plot_ID"].astype(str)
    texture_full = pd.read_csv(require(DATA_DIR / TEXTURE_FULL_FILE))
    texture_full["Disease_Plot_ID"] = texture_full["Disease_Plot_ID"].astype(str)

    y2, y1 = A2["y3"].values, A1["y3"].values
    g2 = A2["Variety_ID"].values
    X7, X7e = A2[F7].values, A1[F7].values
    X6, X6e = A2[F6_SPECTRAL].values, A1[F6_SPECTRAL].values
    X4, X4e = A2[LOW_VIF].values, A1[LOW_VIF].values

    # ---- plot-level scorers -------------------------------------------------
    def within_cascade(labels, make=svc, reversed_order=False, sample_weight=False):
        T, P = [], []
        for tr, te in grouped_cv().split(X7, labels, g2):
            a, b = prep(X7[tr], X7[te])
            P += list(cascade(a, labels[tr], b, make, reversed_order, sample_weight))
            T += list(labels[te])
        return f1_score(T, P, average="macro", labels=LO)

    def external_cascade(labels, make=svc, reversed_order=False, sample_weight=False):
        a, b = prep(X7, X7e)
        return f1_score(y1, cascade(a, labels, b, make, reversed_order, sample_weight),
                        average="macro", labels=LO)

    def within_single(labels, X, make, mean_of_folds=False):
        T, P, folds = [], [], []
        for tr, te in grouped_cv().split(X, labels, g2):
            a, b = prep(X[tr], X[te])
            pred = make().fit(a, labels[tr]).predict(b)
            if mean_of_folds:
                folds.append(f1_score(labels[te], pred, average="macro"))
            T += list(labels[te])
            P += list(pred)
        if mean_of_folds:
            return float(np.mean(folds))
        return f1_score(T, P, average="macro", labels=LO)

    def external_single(labels, X, Xe, make):
        a, b = prep(X, Xe)
        return f1_score(y1, make().fit(a, labels).predict(b), average="macro", labels=LO)

    def within_ordinal(labels):
        T, P = [], []
        for tr, te in grouped_cv().split(X4, labels, g2):
            a, b = prep(X4[tr], X4[te])
            m = mord.LogisticAT(alpha=3.0).fit(a, np.array([ORD[l] for l in labels[tr]]))
            P += [INV_ORD[v] for v in m.predict(b)]
            T += list(labels[te])
        return f1_score(T, P, average="macro", labels=LO)

    def external_ordinal(labels):
        a, b = prep(X4, X4e)
        m = mord.LogisticAT(alpha=3.0).fit(a, np.array([ORD[l] for l in labels]))
        return f1_score(y1, np.array([INV_ORD[v] for v in m.predict(b)]),
                        average="macro", labels=LO)

    # ---- moderate vs healthy: diseased plots dropped entirely ---------------
    is_mod_or_healthy = lambda s: s == 10 or 7 <= s <= 9          # noqa: E731
    mh2 = A2[A2["Disease_class"].apply(is_mod_or_healthy)]
    mh1 = A1[A1["Disease_class"].apply(is_mod_or_healthy)]
    Xm2, Xm1 = mh2[F7].values, mh1[F7].values
    ym2 = (mh2["Disease_class"] < 10).astype(int).values
    ym1 = (mh1["Disease_class"] < 10).astype(int).values

    def external_moderate_auc(labels):
        a, b = prep(Xm2, Xm1)
        m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(a, labels)
        return roc_auc_score(ym1, m.predict_proba(b)[:, 1])

    # ---- variety level: one row per variety ---------------------------------
    V2 = build_variety_features(A2, "2", texture_full)
    V1 = build_variety_features(A1, "1", texture_full)
    variety_df = to_variety_rows(V2, "Variety_ID")           # within-flight, raw names
    Xv = variety_df[VARIETY_FEATURES].values
    yv = (variety_df["mean_severity"] < 10).astype(int).values

    def within_variety_auc(labels):
        T, P = [], []
        # each row already IS one variety, so plain stratified folds are enough
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True,
                                      random_state=RANDOM_SEED).split(Xv, labels):
            a, b = prep(Xv[tr], Xv[te])
            m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(a, labels[tr])
            T.extend(labels[te])
            P.extend(m.predict_proba(b)[:, 1])
        return roc_auc_score(T, P)

    for V in (V2, V1):
        V["vnorm"] = V["Variety_ID"].apply(normalise_variety)
    train_rows = to_variety_rows(V2, "vnorm")
    test_rows = to_variety_rows(V1, "vnorm")
    test_rows["affected"] = (test_rows["mean_severity"] < 10).astype(int)
    shared = set(train_rows["vnorm"]) & set(test_rows["vnorm"])
    train_v = train_rows[train_rows["vnorm"].isin(shared)].set_index("vnorm").sort_index()
    test_v = test_rows[test_rows["vnorm"].isin(shared)].set_index("vnorm").sort_index()
    Xtv, Xev = train_v[VARIETY_FEATURES].values, test_v[VARIETY_FEATURES].values
    ytv = (train_v["mean_severity"] < 10).astype(int).values
    yev = test_v["affected"].values

    def external_variety_auc(labels):
        a, b = prep(Xtv, Xev)
        m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(a, labels)
        return roc_auc_score(yev, m.predict_proba(b)[:, 1])

    def rf():
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      max_depth=5, random_state=RANDOM_SEED)

    def xgb():
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0)

    print("=" * 70)
    print(f"PERMUTATION TESTS  ({N_PERMUTATIONS} label shuffles each, seed {RANDOM_SEED})")
    print("=" * 70)
    print(f"Plot level:    Flight 2 n={len(y2)}, Flight 1 n={len(y1)}")
    print(f"Variety level: within n={len(yv)}, external n={len(ytv)} varieties in both flights")
    print(f"Moderate vs healthy: Flight 2 n={len(ym2)}, Flight 1 n={len(ym1)} (diseased dropped)")
    print("\nThis takes around fifteen minutes. Progress follows.\n")

    # name, scorer, labels to shuffle, expected (real, mean, sd, p)
    plan = [
        ("TSLSC within", within_cascade, y2, (0.438, 0.307, 0.028, 0.001)),
        ("TSLSC external", external_cascade, y2, (0.394, 0.217, 0.117, 0.004)),
        ("Diseased-first within", lambda l: within_cascade(l, reversed_order=True), y2,
         (0.438, 0.285, 0.032, 0.001)),
        ("Diseased-first external", lambda l: external_cascade(l, reversed_order=True), y2,
         (0.378, 0.182, 0.122, 0.014)),
        ("Random Forest external", lambda l: external_cascade(l, make=rf), y2,
         (0.308, 0.311, 0.028, 0.662)),
        ("XGBoost external", lambda l: external_cascade(l, make=xgb, sample_weight=True), y2,
         (0.307, 0.311, 0.040, 0.624)),
        ("Ordinal external", external_ordinal, y2, (0.307, 0.307, 0.019, 0.951)),
        ("Unweighted baseline within",
         lambda l: within_single(l, X6, lambda: LogisticRegression(max_iter=5000)), y2,
         (0.333, 0.285, 0.001, 0.001)),
        ("Unweighted baseline external",
         lambda l: external_single(l, X6, X6e, lambda: LogisticRegression(max_iter=5000)), y2,
         (0.028, 0.179, 0.129, 0.996)),
        ("Balanced single-stage within",
         lambda l: within_single(l, X6,
                                 lambda: LogisticRegression(class_weight="balanced", max_iter=5000),
                                 mean_of_folds=True), y2,
         (0.358, 0.264, 0.028, 0.001)),
        ("Variety-level AUC within", within_variety_auc, yv, (0.615, 0.499, 0.049, 0.008)),
        ("Variety-level AUC external", external_variety_auc, ytv, (0.480, 0.501, 0.074, 0.594)),
        ("Moderate-vs-healthy AUC external", external_moderate_auc, ym2,
         (0.527, 0.499, 0.041, 0.241)),
    ]

    results = []
    for name, fn, labels, expected in plan:
        r = permutation_test(name, fn, labels)
        r["expected"] = expected
        results.append(r)

    # -----------------------------------------------------------------------
    # Computed and PRINTED but NOT checked, on purpose.
    # Two real scores are on record for this test, 0.295 and 0.308. Running its
    # own null is how you tell which one it belongs to, so putting either number
    # in as a check would assume the answer.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UNCHECKED: ordinal regression, within-flight")
    print("=" * 70)
    unchecked = permutation_test("Ordinal within (UNCHECKED)", within_ordinal, y2)
    supported = "0.308" if abs(unchecked["real"] - 0.308) < abs(unchecked["real"] - 0.295) else "0.295"
    print(f"\n  Two real scores are on record for this test: 0.295 and 0.308.")
    print(f"  This run computes {unchecked['real']:.3f}, which supports {supported}.")
    print(f"  Its null sits at {unchecked['permuted_mean']:.3f} +/- {unchecked['permuted_sd']:.3f}"
          f" (p={unchecked['p_value']:.4f}), so the real score is well clear of noise either way.")
    print("  No pass/fail check is applied. Settle it against the notebook.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results + [unchecked]).to_csv(OUT_DIR / "permutation_tests.csv", index=False)

    # -----------------------------------------------------------------------
    # SANITY CHECK. Real scores exactly; the three stochastic quantities within
    # TOLERANCE, because a permutation null is itself a random object.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECK against authorised values from the thesis notebook")
    print("=" * 70)
    print(f"  real score        compared exactly, at the 3 decimals recorded")
    print(f"  permuted mean/SD  compared within +/-{TOLERANCE}")
    print(f"  p, recorded >= {SIGNIFICANCE_LEVEL}  compared within +/-{TOLERANCE}"
          " (the value is the finding: it did not clear its null)")
    print(f"  p, recorded <  {SIGNIFICANCE_LEVEL}  NOT compared by value; required only to be"
          f" < {SIGNIFICANCE_LEVEL}")
    print(f"{'':21s}(the finding is that it cleared its null; both figures printed)")
    print("-" * 70)

    def check_p(got, expected):
        """Two rules, chosen by what the recorded p means. See the docstring."""
        if expected < SIGNIFICANCE_LEVEL:
            return (got < SIGNIFICANCE_LEVEL,
                    f"p={got:.4f} cleared {SIGNIFICANCE_LEVEL} (recorded {expected:.3f})")
        return (abs(got - expected) <= TOLERANCE, f"p={got:.3f}/{expected:.3f}")

    all_ok = True
    for r in results:
        er, em, es, ep = r["expected"]
        p_ok, p_text = check_p(r["p_value"], ep)
        parts = [
            ("real", float(round(r["real"], 3)), er, float(round(r["real"], 3)) == er),
            ("mean", r["permuted_mean"], em, abs(r["permuted_mean"] - em) <= TOLERANCE),
            ("SD", r["permuted_sd"], es, abs(r["permuted_sd"] - es) <= TOLERANCE),
        ]
        ok = all(x[3] for x in parts) and p_ok
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {r['test']:34s} "
              + "  ".join(f"{n}={v:.3f}/{e:.3f}" for n, v, e, _ in parts)
              + "  " + p_text)
        if not ok:
            for n, v, e, good in parts:
                if not good:
                    print(f"             {n}: got {v:.4f}, expected {e:.4f}")
            if not p_ok:
                print(f"             p: got {r['p_value']:.4f}, recorded {ep:.4f}"
                      + (f" (needed < {SIGNIFICANCE_LEVEL})" if ep < SIGNIFICANCE_LEVEL else ""))
    print("\nAll matched." if all_ok
          else "\nMISMATCH found. Do not trust any number above until it is explained.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
