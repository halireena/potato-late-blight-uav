"""
03_architectures.py
===================
Runs the comparator architectures the locked model was chosen over, so the
choice can be seen rather than taken on trust.

Run from the repo root, after 01_build_plot_table.py:
    python src/03_architectures.py

The idea in plain words
-----------------------
02_locked_model.py shows what the chosen model does. It does not show why it
was chosen. This script trains the alternatives that were tried and rejected,
on the same plots, the same features and the same splits, and reports each one
under both conditions:

    within-flight  5-fold StratifiedGroupKFold on Flight 2, grouped by variety
    external       fit once on all of Flight 2, predict Flight 1

The point is the gap between those two columns. Several alternatives beat the
locked cascade within-flight and then collapse on the external flight, landing
exactly on the majority-class floor: the score you get by calling every plot
healthy and doing no work at all. That floor is computed here as a yardstick,
because a macro-F1 near it means the model found nothing.

Architectures, in the order printed
-----------------------------------
0. Majority-class floor          predict healthy for everything. Not a model.
1. Locked two-stage SVM cascade  the chosen model, repeated here as reference.
2. Unweighted single model       one 3-class logistic regression, no class
                                 weighting. Shows what ignoring the class
                                 imbalance costs.
3. Ordinal regression            mord.LogisticAT. Severity is ordered, so this
                                 ought to fit; it does not.
4. Random Forest cascade         same cascade, trees instead of a linear SVM.
5. XGBoost cascade               same again, gradient boosting.
6. Three-stage cascade           healthy first, then moderate vs diseased among
                                 what is left.
7. Single-stage three-class SVC  the same SVM, asked for all three classes at
                                 once instead of in two steps.
8. Diseased-first cascade        the locked cascade with its two stages
                                 swapped, so diseased is decided first.
9. MLP cascade                   a small neural network in the cascade.
10. One-vs-rest ensemble         three independent binary models, one per class,
                                 and whichever is most confident wins. No
                                 cascade, no ordering, everything decided at once.

A warning about architecture 10
-------------------------------
The one-vs-rest ensemble needs predicted probabilities, so its three SVMs are
fitted with probability=True. That makes sklearn run an internal 5-fold Platt
calibration, and the notebook left it unseeded. Its within-flight score is
therefore a lottery: ten unseeded runs here gave 0.300, 0.307, 0.308, 0.315 and
0.323, with 0.315 coming up twice. That is where the 0.308-versus-0.315 dispute
comes from -- not two methods, one method run twice.

It is seeded with RANDOM_SEED here, because this repository promises that a
rerun reproduces, and an unseeded check would pass or fail at random. Seeding
fixes it at one value; it does not make that value the recorded one. So the
within-flight score is REPORTED beside the notebook's 0.315 and is NOT checked:
the recorded procedure does not determine it. The test-flight score is stable
at 0.308 either way, and that one is checked.

Two feature sets are used, and this is not an oversight
-------------------------------------------------------
Most architectures use the locked seven features. Two do not, because the
numbers recorded for them in the notebook were produced on a different set,
and reproducing a number means running what actually produced it:

    ordinal regression      the four low-VIF spectral means only
    unweighted baseline     six spectral means, no texture

Both are flagged in the printed output. The notebook discloses the ordinal
one as a limitation; the unweighted one is recorded here for the first time.

Expected values
---------------
Every expected value in the sanity check was printed by an executed cell of
the thesis notebook (mine_ML.ipynb) and was supplied as authorised. No value
is invented, guessed or derived here. Two quantities that the notebook records
inconsistently are printed WITHOUT a pass/fail check and are listed in
HANDOVER.md as unresolved. The script exits with code 1 on any mismatch.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import mord

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

# Feature sets, all column names from the processed tables (no data values).
LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
TEX_N = ["NIR_max_norm", "RedEdge_p90_norm", "Red_min_norm"]
F7 = LOW_VIF + TEX_N                       # the locked seven
F6_UNWEIGHTED = ["canopy_cover_mean", "NDVI_mean", "NDRE_mean",
                 "GNDVI_mean", "EVI2_mean", "OSAVI_mean"]

LO = ["diseased", "moderate", "healthy"]   # label order for every matrix and metric
ORD = {"diseased": 0, "moderate": 1, "healthy": 2}
INV_ORD = {v: k for k, v in ORD.items()}


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def svc():
    return SVC(kernel="linear", class_weight="balanced", C=0.1)


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  max_depth=5, random_state=RANDOM_SEED)


def xgb():
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0)


def mlp():
    return MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000, random_state=RANDOM_SEED)


def fit_scale(Xtr, Xte):
    """Median-impute then standardise, both fitted on the training side only."""
    im = SimpleImputer(strategy="median")
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def combine(a, b):
    """Healthy-first cascade rule: Stage A, then Stage B, else moderate."""
    return np.array(["healthy" if i == 1 else ("diseased" if j == 1 else "moderate")
                     for i, j in zip(a, b)])


def combine_diseased_first(a, b):
    """Reversed cascade rule: diseased decided first, then healthy."""
    return np.array(["diseased" if i == 1 else ("healthy" if j == 1 else "moderate")
                     for i, j in zip(a, b)])


# ---------------------------------------------------------------------------
# One entry point per architecture: each returns predictions for one condition
# ---------------------------------------------------------------------------
def cascade(Xtr, ytr, Xte, make, reversed_order=False, sample_weight=False):
    """Two-stage cascade, already-scaled inputs."""
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
    rule = combine_diseased_first if reversed_order else combine
    return rule(*preds)


def three_stage(Xtr, ytr, Xte):
    """Stage A healthy-vs-rest; Stage C moderate-vs-diseased, trained only on
    the not-healthy training plots."""
    pa = svc().fit(Xtr, (ytr == "healthy").astype(int)).predict(Xte)
    nh = ytr != "healthy"
    mc = svc().fit(Xtr[nh], (ytr[nh] == "moderate").astype(int))
    pc = mc.predict(Xte)
    return np.array(["healthy" if a == 1 else ("moderate" if c == 1 else "diseased")
                     for a, c in zip(pa, pc)])


def single_stage(Xtr, ytr, Xte, make):
    """One model asked for all three classes directly."""
    return make().fit(Xtr, ytr).predict(Xte)


def one_vs_rest(Xtr, ytr, Xte):
    """Three independent binary models, one per class; most confident wins.

    Seeded, unlike the notebook: probability=True runs an internal randomised
    calibration, and leaving it unseeded makes the within-flight score vary by
    about 0.02 between runs. See the warning in the module docstring."""
    probs = {}
    for c in LO:
        m = SVC(kernel="linear", class_weight="balanced", C=0.1,
                probability=True, random_state=RANDOM_SEED)
        probs[c] = m.fit(Xtr, (ytr == c).astype(int)).predict_proba(Xte)[:, 1]
    return np.array([LO[int(np.argmax([probs[c][i] for c in LO]))]
                     for i in range(len(Xte))])


def ordinal(Xtr, ytr, Xte):
    """Ordinal regression on integer-coded severity, decoded back to labels."""
    m = mord.LogisticAT(alpha=3.0).fit(Xtr, np.array([ORD[l] for l in ytr]))
    return np.array([INV_ORD[v] for v in m.predict(Xte)])


# ---------------------------------------------------------------------------
def run_both(name, predict_fn, A2, A1, feats, cv):
    """Run one architecture within-flight and externally; print and return both."""
    X2, X1 = A2[feats].values, A1[feats].values
    y2, y1 = A2["y3"].values, A1["y3"].values
    g2 = A2["Variety_ID"].values

    Tw, Pw = [], []
    for tr, te in cv.split(X2, y2, g2):
        Xtr, Xte = fit_scale(X2[tr], X2[te])
        Pw += list(predict_fn(Xtr, y2[tr], Xte))
        Tw += list(y2[te])
    Tw, Pw = np.array(Tw), np.array(Pw)

    X2s, X1s = fit_scale(X2, X1)
    Pe = np.array(predict_fn(X2s, y2, X1s))

    out = {"architecture": name, "n_features": len(feats)}
    for cond, T, P in (("within", Tw, Pw), ("external", y1, Pe)):
        dm = T == "diseased"
        out[f"{cond}_f1"] = f1_score(T, P, average="macro", labels=LO)
        out[f"{cond}_qwk"] = cohen_kappa_score([ORD[t] for t in T], [ORD[p] for p in P],
                                               weights="quadratic")
        out[f"{cond}_dis_recalled"] = int((P[dm] == "diseased").sum())
        out[f"{cond}_dis_total"] = int(dm.sum())

    print(f"\n{name}  ({len(feats)} features)")
    for cond in ("within", "external"):
        print(f"  {cond:8s} macro-F1={out[f'{cond}_f1']:.3f}  QWK={out[f'{cond}_qwk']:.3f}  "
              f"diseased recall={out[f'{cond}_dis_recalled']}/{out[f'{cond}_dis_total']}")
    print("  external confusion matrix (rows = true; diseased, moderate, healthy)")
    for label, row in zip(LO, confusion_matrix(y1, Pe, labels=LO)):
        print(f"    {label:9s} {row}")
    return out


def main():
    A2 = pd.read_csv(require(PROCESSED_DIR / "A2_clean_tex_norm.csv"))   # Flight 2, training
    A1 = pd.read_csv(require(PROCESSED_DIR / "A1_clean_tex_norm.csv"))   # Flight 1, test
    for d in (A2, A1):
        d["y3"] = d["Disease_class"].apply(lab)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print("=" * 70)
    print("COMPARATOR ARCHITECTURES, within-flight CV and external flight")
    print("=" * 70)

    # 0. Majority-class floor: no model, no fitting, no features ------------
    y1 = A1["y3"].values
    y2 = A2["y3"].values
    floor_ext = f1_score(y1, np.array(["healthy"] * len(y1)), average="macro", labels=LO)
    floor_win = f1_score(y2, np.array(["healthy"] * len(y2)), average="macro", labels=LO)
    print("\nMajority-class floor (predict healthy for every plot; not a model)")
    print(f"  within   macro-F1={floor_win:.3f}")
    print(f"  external macro-F1={floor_ext:.3f}   <- any model at this value found nothing")

    rows = []
    rows.append(run_both("Locked two-stage SVM cascade",
                         lambda a, b, c: cascade(a, b, c, svc), A2, A1, F7, cv))
    rows.append(run_both("Unweighted single model (6 spectral means, no texture)",
                         lambda a, b, c: single_stage(a, b, c, lambda: LogisticRegression(max_iter=5000)),
                         A2, A1, F6_UNWEIGHTED, cv))
    rows.append(run_both("Ordinal regression, mord.LogisticAT (4 low-VIF features)",
                         ordinal, A2, A1, LOW_VIF, cv))
    rows.append(run_both("Random Forest cascade",
                         lambda a, b, c: cascade(a, b, c, rf), A2, A1, F7, cv))
    rows.append(run_both("XGBoost cascade",
                         lambda a, b, c: cascade(a, b, c, xgb, sample_weight=True), A2, A1, F7, cv))
    rows.append(run_both("Three-stage cascade", three_stage, A2, A1, F7, cv))
    rows.append(run_both("Single-stage three-class SVC",
                         lambda a, b, c: single_stage(a, b, c, svc), A2, A1, F7, cv))
    rows.append(run_both("Diseased-first (reversed) cascade",
                         lambda a, b, c: cascade(a, b, c, svc, reversed_order=True), A2, A1, F7, cv))
    rows.append(run_both("MLP cascade (8 hidden units)",
                         lambda a, b, c: cascade(a, b, c, mlp), A2, A1, F7, cv))
    rows.append(run_both("One-vs-rest ensemble (most confident wins)",
                         one_vs_rest, A2, A1, F7, cv))

    by_name = {r["architecture"]: r for r in rows}
    summary = pd.DataFrame(
        [{"architecture": "Majority-class floor", "n_features": 0,
          "within_f1": floor_win, "external_f1": floor_ext}] + rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "architecture_comparison.csv", index=False)

    # -----------------------------------------------------------------------
    # Printed but NOT checked: the notebook records two different values for
    # each of these, so there is no single value to check against. Reported as
    # unresolved in HANDOVER.md for the author to settle against the notebook.
    # -----------------------------------------------------------------------
    LOCK = "Locked two-stage SVM cascade"
    UNW = "Unweighted single model (6 spectral means, no texture)"
    ORDN = "Ordinal regression, mord.LogisticAT (4 low-VIF features)"
    RFC = "Random Forest cascade"
    XGBC = "XGBoost cascade"
    TS3 = "Three-stage cascade"
    SS = "Single-stage three-class SVC"
    DF = "Diseased-first (reversed) cascade"
    MLPC = "MLP cascade (8 hidden units)"
    OVR = "One-vs-rest ensemble (most confident wins)"

    ordv = by_name[ORDN]
    print("\n" + "=" * 70)
    print("PRINTED BUT NOT CHECKED (the notebook disagrees with itself)")
    print("=" * 70)
    print(f"  Ordinal regression, within-flight macro-F1 = {ordv['within_f1']:.3f}")
    print("    two sources recorded for this: 0.308 and 0.295. Unresolved, so no check.")
    print(f"  One-vs-rest ensemble, within-flight macro-F1")
    print(f"    computed here     {by_name[OVR]['within_f1']:.3f}   (seeded with RANDOM_SEED)")
    print(f"    notebook records  0.315")
    print("    NOT CHECKED. The notebook fits these three models with probability=True and")
    print("    leaves the internal Platt calibration unseeded, so the value is not determined")
    print("    by the recorded procedure. Ten unseeded runs of this exact computation spanned")
    print("    0.300 to 0.323, taking the values 0.300, 0.307, 0.308, 0.315 and 0.323, with")
    print("    0.315 coming up twice in ten. Both numbers on record, 0.308 and 0.315, are")
    print("    draws from that same spread. It is seeded here so this repository stays")
    print("    deterministic, which fixes it at one value rather than making it correct.")
    print("    The TEST-flight score is stable at 0.308 and is checked.")
    print("\n  MLP cascade: checked, at 0.291 within and 0.308 external.")

    # -----------------------------------------------------------------------
    # SANITY CHECK: every expected value below was printed by an executed cell
    # of mine_ML.ipynb and supplied as authorised. Floats are compared at the
    # 3 decimal places that were printed.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY CHECK against authorised values from the thesis notebook")
    print("=" * 70)

    def f1(name, cond):
        return float(round(by_name[name][f"{cond}_f1"], 3))

    def qwk(name, cond):
        return float(round(by_name[name][f"{cond}_qwk"], 3))

    def rec(name, cond):
        return (by_name[name][f"{cond}_dis_recalled"], by_name[name][f"{cond}_dis_total"])


    checks = [
        ("Majority-class floor, external macro-F1", float(round(floor_ext, 3)), 0.308),

        ("Locked cascade, within macro-F1", f1(LOCK, "within"), 0.438),
        ("Locked cascade, within QWK", qwk(LOCK, "within"), 0.206),
        ("Locked cascade, within diseased recall", rec(LOCK, "within"), (53, 86)),
        ("Locked cascade, external macro-F1", f1(LOCK, "external"), 0.394),
        ("Locked cascade, external QWK", qwk(LOCK, "external"), 0.206),
        ("Locked cascade, external diseased recall", rec(LOCK, "external"), (8, 25)),

        ("Unweighted baseline, within macro-F1", f1(UNW, "within"), 0.333),
        ("Unweighted baseline, external macro-F1", f1(UNW, "external"), 0.028),

        ("Ordinal regression, external macro-F1", f1(ORDN, "external"), 0.307),

        ("Random Forest cascade, external macro-F1", f1(RFC, "external"), 0.308),
        ("XGBoost cascade, external macro-F1", f1(XGBC, "external"), 0.307),

        ("Three-stage cascade, within macro-F1", f1(TS3, "within"), 0.446),
        ("Three-stage cascade, within diseased recall", rec(TS3, "within"), (47, 86)),
        ("Three-stage cascade, external macro-F1", f1(TS3, "external"), 0.381),
        ("Three-stage cascade, external diseased recall", rec(TS3, "external"), (4, 25)),

        ("Single-stage SVC, within macro-F1", f1(SS, "within"), 0.380),
        ("Single-stage SVC, within diseased recall", rec(SS, "within"), (51, 86)),
        ("Single-stage SVC, external macro-F1", f1(SS, "external"), 0.403),
        ("Single-stage SVC, external diseased recall", rec(SS, "external"), (8, 25)),

        ("Diseased-first cascade, within macro-F1", f1(DF, "within"), 0.438),
        ("Diseased-first cascade, within QWK", qwk(DF, "within"), 0.211),
        ("Diseased-first cascade, external macro-F1", f1(DF, "external"), 0.378),
        ("Diseased-first cascade, external diseased recall", rec(DF, "external"), (13, 25)),

        ("MLP cascade, within macro-F1", f1(MLPC, "within"), 0.291),
        ("MLP cascade, external macro-F1", f1(MLPC, "external"), 0.308),

        # One-vs-rest WITHIN-flight is NOT checked. The notebook leaves its Platt
        # calibration unseeded, so the recorded procedure does not determine the
        # value. Reported beside 0.315 in the printed-not-checked section instead.
        ("One-vs-rest, external macro-F1", f1(OVR, "external"), 0.308),
    ]

    all_ok = True
    for name, got, expected in checks:
        ok = got == expected
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name:52s} got {got}  expected {expected}")
    print("\nAll matched." if all_ok
          else "\nMISMATCH found. Do not trust any number above until it is explained.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
