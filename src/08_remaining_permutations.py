"""
08_remaining_permutations.py
============================
The rest of Supplementary Table S2: the permutation tests that
04_permutation_tests.py does not run.

Run from the repo root, after 01_build_plot_table.py:
    python src/08_remaining_permutations.py

Same question, same protocol
----------------------------
Identical to 04: 1,000 shuffles of the training labels, RandomState(42), the
splitter rebuilt from whatever labels it is handed, and

    p = (number of permuted scores >= the real score + 1) / (N + 1)

Two rows use 200 shuffles rather than 1,000, because that is what was recorded
for them. Each row prints the count it used.

The eleven rows checked here
----------------------------
    three-stage cascade          within-flight and test flight
    single-stage three-class     test flight
    Random Forest cascade        within-flight
    XGBoost cascade              within-flight
    MLP cascade                  within-flight and test flight  (200 shuffles)
    binary healthy-vs-diseased   within-flight and test flight
    one-vs-rest ensemble         within-flight and test flight

04 already covers the locked cascade, diseased-first, the external collapses of
RF, XGBoost and ordinal, the unweighted baseline, the balanced single-stage
model, the variety-level tests and moderate-vs-healthy. Between them the two
scripts cover the whole table.

How the numbers are checked
---------------------------
Exactly as in 04, and for the same reasons, which are argued there in full:
real scores exactly at three decimals; permuted means and SDs within +/-0.02;
p compared by value only where the recorded p is 0.05 or above, and otherwise
required only to be below 0.05, because with 1,000 shuffles the smallest
reachable p is 1/1001 and the exact figure wanders between runs.

Two rows will not behave
------------------------
The one-vs-rest rows depend on probability=True, which runs an internal
randomised calibration that the notebook left unseeded. Its within-flight score
is a lottery across roughly 0.300 to 0.323. It is seeded here so this script is
deterministic, which means its real score is a fixed number and not necessarily
the one on record. See the docstring of 03_architectures.py.

The twelfth row is disputed and is NOT checked
----------------------------------------------
Supplementary Table S2 records a single-stage three-class within-flight score of
0.380 against a null of 0.263 +/- 0.034. 04 verifies a balanced single-stage
model at 0.358 against a null of 0.264 +/- 0.028. Those are two different models
on two different feature sets aggregated two different ways, not one number
recorded twice. Both are computed here, both are printed with their feature set
and aggregation named, and each observed score is read against both nulls. No
pass/fail check is applied to either.

This script is slow: about seventy minutes, most of it the Random Forest and
XGBoost within-flight nulls, which fit ten forests per shuffle. Progress is
printed as it goes.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

# See 04_permutation_tests.py for why this is silenced and nothing else is.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*converge.*")

N_DEFAULT = 1000
N_MLP = 200
TOLERANCE = 0.02
SIGNIFICANCE_LEVEL = 0.05

LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
TEX_N = ["NIR_max_norm", "RedEdge_p90_norm", "Red_min_norm"]
F7 = LOW_VIF + TEX_N
F6_SPECTRAL = ["canopy_cover_mean", "NDVI_mean", "NDRE_mean",
               "GNDVI_mean", "EVI2_mean", "OSAVI_mean"]
LO = ["diseased", "moderate", "healthy"]


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def prep(Xtr, Xte):
    im = SimpleImputer(strategy="median")
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def svc(probability=False):
    return SVC(kernel="linear", class_weight="balanced", C=0.1,
               probability=probability,
               random_state=RANDOM_SEED if probability else None)


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  max_depth=5, random_state=RANDOM_SEED)


def xgb():
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=RANDOM_SEED, verbosity=0)


def mlp():
    return MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000, random_state=RANDOM_SEED)


def grouped_cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def cascade(Xtr, ytr, Xte, make, sample_weight=False):
    preds = []
    for target in ("healthy", "diseased"):
        yb = (ytr == target).astype(int)
        m = make()
        if sample_weight:
            m.fit(Xtr, yb, sample_weight=compute_sample_weight("balanced", yb))
        else:
            m.fit(Xtr, yb)
        preds.append(m.predict(Xte))
    a, b = preds
    return np.array(["healthy" if i == 1 else ("diseased" if j == 1 else "moderate")
                     for i, j in zip(a, b)])


def three_stage(Xtr, ytr, Xte):
    pa = svc().fit(Xtr, (ytr == "healthy").astype(int)).predict(Xte)
    nh = ytr != "healthy"
    pc = svc().fit(Xtr[nh], (ytr[nh] == "moderate").astype(int)).predict(Xte)
    return np.array(["healthy" if a == 1 else ("moderate" if c == 1 else "diseased")
                     for a, c in zip(pa, pc)])


def one_vs_rest(Xtr, ytr, Xte):
    """Three binary models, most confident wins. Seeded; see the docstring."""
    probs = {}
    for c in LO:
        probs[c] = svc(probability=True).fit(
            Xtr, (ytr == c).astype(int)).predict_proba(Xte)[:, 1]
    return np.array([LO[int(np.argmax([probs[c][i] for c in LO]))]
                     for i in range(len(Xte))])


def permutation_test(name, score_fn, y_real, n_perm):
    real = score_fn(y_real)
    rng = np.random.RandomState(RANDOM_SEED)
    perm = np.empty(n_perm)
    step = max(1, n_perm // 4)
    for i in range(n_perm):
        perm[i] = score_fn(rng.permutation(y_real))
        if (i + 1) % step == 0:
            print(f"    {name}: {i + 1}/{n_perm}", flush=True)
    p = (np.sum(perm >= real) + 1) / (n_perm + 1)
    print(f"  {name}: real={real:.3f}  permuted mean={perm.mean():.3f}"
          f" +/- {perm.std():.3f}  p={p:.4f}  ({n_perm} shuffles)", flush=True)
    return {"test": name, "real": real, "permuted_mean": float(perm.mean()),
            "permuted_sd": float(perm.std()), "p_value": float(p), "n_perm": n_perm}


def main():
    A2 = pd.read_csv(require(PROCESSED_DIR / "A2_clean_tex_norm.csv"))
    A1 = pd.read_csv(require(PROCESSED_DIR / "A1_clean_tex_norm.csv"))
    for d in (A2, A1):
        d["y3"] = d["Disease_class"].apply(lab)
    y2, y1 = A2["y3"].values, A1["y3"].values
    g2 = A2["Variety_ID"].values
    X7, X7e = A2[F7].values, A1[F7].values
    X6 = A2[F6_SPECTRAL].values

    # binary subsets: moderate dropped entirely
    keep2, keep1 = y2 != "moderate", y1 != "moderate"
    Xb2, Xb1 = X7[keep2], X7e[keep1]
    yb2 = (y2[keep2] == "diseased").astype(int)
    yb1 = (y1[keep1] == "diseased").astype(int)
    gb2 = g2[keep2]

    def within(labels, predict_fn, X=None, groups=None, y_for_split=None):
        X = X7 if X is None else X
        groups = g2 if groups is None else groups
        T, P = [], []
        for tr, te in grouped_cv().split(X, labels, groups):
            a, b = prep(X[tr], X[te])
            P += list(predict_fn(a, labels[tr], b))
            T += list(labels[te])
        return T, P

    def wf_macro(labels, predict_fn):
        T, P = within(labels, predict_fn)
        return f1_score(T, P, average="macro", labels=LO)

    def ext_macro(labels, predict_fn):
        a, b = prep(X7, X7e)
        return f1_score(y1, predict_fn(a, labels, b), average="macro", labels=LO)

    def wf_binary(labels):
        T, P = [], []
        for tr, te in grouped_cv().split(Xb2, labels, gb2):
            a, b = prep(Xb2[tr], Xb2[te])
            P += list(svc().fit(a, labels[tr]).predict(b))
            T += list(labels[te])
        return f1_score(T, P, average="macro")

    def ext_binary(labels):
        a, b = prep(Xb2, Xb1)
        return f1_score(yb1, svc().fit(a, labels).predict(b), average="macro")

    print("=" * 74)
    print(f"REMAINING PERMUTATION TESTS  (seed {RANDOM_SEED})")
    print("=" * 74)
    print(f"Plot level: Flight 2 n={len(y2)}, Flight 1 n={len(y1)}")
    print(f"Binary subsets (moderate dropped): Flight 2 n={len(yb2)}, Flight 1 n={len(yb1)}")
    print("\nAbout seventy minutes. Progress follows.\n")

    plan = [
        ("Three-stage within", lambda l: wf_macro(l, three_stage), y2, N_DEFAULT,
         (0.446, 0.306, 0.029, 0.001)),
        ("Three-stage test", lambda l: ext_macro(l, three_stage), y2, N_DEFAULT,
         (0.381, 0.222, 0.109, 0.009)),
        ("Single-stage test", lambda l: ext_macro(l, lambda a, b, c: svc().fit(a, b).predict(c)),
         y2, N_DEFAULT, (0.403, 0.165, 0.111, 0.003)),
        ("Random Forest within", lambda l: wf_macro(l, lambda a, b, c: cascade(a, b, c, rf)),
         y2, N_DEFAULT, (0.357, 0.321, 0.022, 0.061)),
        ("XGBoost within",
         lambda l: wf_macro(l, lambda a, b, c: cascade(a, b, c, xgb, sample_weight=True)),
         y2, N_DEFAULT, (0.361, 0.324, 0.022, 0.053)),
        ("MLP within", lambda l: wf_macro(l, lambda a, b, c: cascade(a, b, c, mlp)),
         y2, N_MLP, (0.291, 0.291, 0.010, 0.413)),
        ("MLP test", lambda l: ext_macro(l, lambda a, b, c: cascade(a, b, c, mlp)),
         y2, N_MLP, (0.308, 0.311, 0.010, 0.672)),
        ("Binary within", wf_binary, yb2, N_DEFAULT, (0.602, 0.444, 0.040, 0.001)),
        ("Binary test", ext_binary, yb2, N_DEFAULT, (0.617, 0.302, 0.192, 0.001)),
        ("One-vs-rest within", lambda l: wf_macro(l, one_vs_rest), y2, N_DEFAULT,
         (0.315, 0.285, 0.001, 0.001)),
        ("One-vs-rest test", lambda l: ext_macro(l, one_vs_rest), y2, N_DEFAULT,
         (0.308, 0.308, 0.001, 0.983)),
    ]

    results = []
    for name, fn, labels, n_perm, expected in plan:
        r = permutation_test(name, fn, labels, n_perm)
        r["expected"] = expected
        results.append(r)

    # -----------------------------------------------------------------------
    # The disputed twelfth row. Two different models; both computed, neither
    # checked, each score read against both nulls.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("UNCHECKED: single-stage three-class, within-flight  (DISPUTED)")
    print("=" * 74)
    print("Two records exist and they are not the same model:\n")
    print("  Table S2 : single-stage three-class SVC, the locked SEVEN features,")
    print("             predictions POOLED across the five folds")
    print("  04       : single-stage LogisticRegression, class_weight='balanced',")
    print("             SIX spectral means, MEAN of the five per-fold scores\n")

    def svc7_pooled(labels):
        return wf_macro(labels, lambda a, b, c: svc().fit(a, b).predict(c))

    def logreg6_meanfolds(labels):
        folds = []
        for tr, te in grouped_cv().split(X6, labels, g2):
            a, b = prep(X6[tr], X6[te])
            m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(a, labels[tr])
            folds.append(f1_score(labels[te], m.predict(b), average="macro"))
        return float(np.mean(folds))

    disputed = [
        ("SVC, 7 features, pooled", svc7_pooled, 0.380, (0.263, 0.034)),
        ("LogReg balanced, 6 features, mean-of-folds", logreg6_meanfolds, 0.358, (0.264, 0.028)),
    ]
    computed = []
    for name, fn, recorded, null in disputed:
        r = permutation_test(name, fn, y2, N_DEFAULT)
        r["recorded"] = recorded
        r["recorded_null"] = null
        computed.append(r)

    print("\n  Which null is each observed score consistent with?")
    for r in computed:
        print(f"\n    {r['test']}")
        print(f"      observed {r['real']:.3f}  (recorded {r['recorded']:.3f})")
        print(f"      its own null: {r['permuted_mean']:.3f} +/- {r['permuted_sd']:.3f}"
              f"  (recorded {r['recorded_null'][0]:.3f} +/- {r['recorded_null'][1]:.3f})")
        for other in computed:
            m, sd = other["permuted_mean"], other["permuted_sd"]
            z = (r["real"] - m) / sd if sd else float("inf")
            verdict = "INSIDE" if abs(z) <= 2 else "outside"
            print(f"      against the {other['test']} null "
                  f"({m:.3f} +/- {sd:.3f}): {z:+.1f} SD, {verdict} 2 SD")
    print("\n  Both nulls sit near the majority-class floor, so both observed scores clear")
    print("  both nulls. The nulls do NOT separate the two records: the difference between")
    print("  0.380 and 0.358 is the model, the feature set and the aggregation, not the")
    print("  significance. No pass/fail check is applied to either.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results + computed).to_csv(OUT_DIR / "remaining_permutations.csv", index=False)

    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("SANITY CHECK against authorised values from Supplementary Table S2")
    print("=" * 74)
    print("  real score        compared exactly, at the 3 decimals recorded")
    print(f"  permuted mean/SD  compared within +/-{TOLERANCE}")
    print(f"  p, recorded >= {SIGNIFICANCE_LEVEL}  compared within +/-{TOLERANCE}")
    print(f"  p, recorded <  {SIGNIFICANCE_LEVEL}  required only to be < {SIGNIFICANCE_LEVEL}")
    print("-" * 74)

    def check_p(got, expected):
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
        print(f"  [{'OK' if ok else 'MISMATCH'}] {r['test']:26s} "
              + "  ".join(f"{n}={v:.3f}/{e:.3f}" for n, v, e, _ in parts) + "  " + p_text)
        if not ok:
            for n, v, e, good in parts:
                if not good:
                    print(f"             {n}: got {v:.4f}, expected {e:.4f}")
            if not p_ok:
                print(f"             p: got {r['p_value']:.4f}, recorded {ep:.4f}")
    print("\nAll matched." if all_ok
          else "\nMISMATCH found. Do not trust any number above until it is explained.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
