"""
02_locked_model.py
==================
Reproduces the locked model, the Two-Stage Linear SVM Cascade (TSLSC), and its
headline numbers.

Run from the repo root, after 01_build_plot_table.py:
    python src/02_locked_model.py

The model in plain words
------------------------
Each plot's 1 to 10 score becomes a class: 10 = healthy, 7 to 9 = moderate,
1 to 6 = diseased. Two yes/no linear SVMs are trained on the same data:
    Stage A: is this plot healthy?
    Stage B: is this plot diseased?
At prediction time, Stage A decides first. If A says healthy, that is the answer.
Otherwise, if B says diseased, the plot is diseased. Anything left is moderate.
Both SVMs use C=0.1 and class_weight='balanced', so mistakes on the rare classes
cost more during training.

Features (seven): four spectral index means with low collinearity, plus three
texture statistics z-scored within each flight.

How it is evaluated
-------------------
1. Within-flight: 5-fold StratifiedGroupKFold on Flight 2, grouped by variety,
   so replicate plots of a variety never sit on both sides of a split. The imputer
   and scaler are fitted on the training folds only.
2. External: fit once on all of Flight 2, predict Flight 1, a temporally
   independent flight the model never saw.
3. Plot overlap between flights, per-stage AUCs, a per-flight standardisation
   test, and a bootstrap confidence interval on the external macro-F1.

Every expected value in the sanity check was printed by cell 390 of the thesis
notebook (my_code.html). The script exits with code 1 on any mismatch.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
TEX_N = ["NIR_max_norm", "RedEdge_p90_norm", "Red_min_norm"]
F7 = LOW_VIF + TEX_N
LO = ["diseased", "moderate", "healthy"]          # label order for every matrix and metric
ORD = {"diseased": 0, "moderate": 1, "healthy": 2}
N_BOOT = 5000


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def mk():
    return SVC(kernel="linear", class_weight="balanced", C=0.1)


def combine(a, b):
    """Cascade rule: Stage A (healthy) first, then Stage B (diseased), else moderate."""
    return np.array(["healthy" if i == 1 else ("diseased" if j == 1 else "moderate") for i, j in zip(a, b)])


def fit_scale(Xtr, Xte):
    im = SimpleImputer(strategy="median")
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def cascade_cv(X, y, g, cv):
    T, P = [], []
    for tr, te in cv.split(X, y, g):
        Xtr, Xte = fit_scale(X[tr], X[te])
        a = mk().fit(Xtr, (y[tr] == "healthy").astype(int)).predict(Xte)
        b = mk().fit(Xtr, (y[tr] == "diseased").astype(int)).predict(Xte)
        P += list(combine(a, b))
        T += list(y[te])
    return np.array(T), np.array(P)


def metrics(T, P):
    f1 = f1_score(T, P, average="macro", labels=LO)
    qwk = cohen_kappa_score([ORD[t] for t in T], [ORD[p] for p in P], weights="quadratic")
    dm = T == "diseased"
    return f1, qwk, int((P[dm] == "diseased").sum()), int(dm.sum())


def show(name, T, P):
    f1, qwk, dr, dn = metrics(T, P)
    print(f"{name}: macro-F1={f1:.3f}  QWK={qwk:.3f}  diseased recall={dr}/{dn}={dr / dn:.3f}")
    print("  confusion matrix (rows = true, columns = predicted; diseased, moderate, healthy)")
    for label, row in zip(LO, confusion_matrix(T, P, labels=LO)):
        print(f"    {label:9s} {row}")
    return f1, qwk, dr, dn


def main():
    A2 = pd.read_csv(require(PROCESSED_DIR / "A2_clean_tex_norm.csv"))   # Flight 2, training
    A1 = pd.read_csv(require(PROCESSED_DIR / "A1_clean_tex_norm.csv"))   # Flight 1, test
    for d in (A2, A1):
        d["y3"] = d["Disease_class"].apply(lab)
        d["pid"] = d["Disease_Plot_ID"].astype(str)
    X2, y2, g2 = A2[F7].values, A2["y3"].values, A2["Variety_ID"].values
    X1, y1 = A1[F7].values, A1["y3"].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 1. Within-flight and external ------------------------------------------
    print("=" * 70 + "\n1. WITHIN-FLIGHT (grouped 5-fold CV) and EXTERNAL (Flight 2 -> Flight 1)\n" + "=" * 70)
    Tw, Pw = cascade_cv(X2, y2, g2, sgkf)
    f1_w, qwk_w, dr_w, dn_w = show("WITHIN-FLIGHT", Tw, Pw)

    X2s, X1s = fit_scale(X2, X1)
    Pe = combine(mk().fit(X2s, (y2 == "healthy").astype(int)).predict(X1s),
                 mk().fit(X2s, (y2 == "diseased").astype(int)).predict(X1s))
    f1_e, qwk_e, dr_e, dn_e = show("EXTERNAL", y1, Pe)

    # 2. Plot overlap between flights -----------------------------------------
    print("\n" + "=" * 70 + "\n2. PLOTS PRESENT IN BOTH FLIGHTS\n" + "=" * 70)
    matched = set(A1["pid"]) & set(A2["pid"])
    m2, m1 = A2.set_index("pid")["y3"], A1.set_index("pid")["y3"]
    same = sum(m2.get(p) == m1.get(p) for p in matched)
    print(f"Matched plots: {len(matched)} of {len(A1)} test-flight plots")
    print(f"Same three-class label in both flights: {same} of {len(matched)}")

    # 3. Per-stage AUC (threshold-free) ---------------------------------------
    print("\n" + "=" * 70 + "\n3. PER-STAGE AUC, within-flight CV vs external\n" + "=" * 70)

    def auc_cv(target):
        scores, truth = [], []
        for tr, te in sgkf.split(X2, y2, g2):
            Xtr, Xte = fit_scale(X2[tr], X2[te])
            scores += list(mk().fit(Xtr, (y2[tr] == target).astype(int)).decision_function(Xte))
            truth += list((y2[te] == target).astype(int))
        return roc_auc_score(truth, scores)

    def auc_ext(target):
        m = mk().fit(X2s, (y2 == target).astype(int))
        return roc_auc_score((y1 == target).astype(int), m.decision_function(X1s))

    auc = {t: (auc_cv(t), auc_ext(t)) for t in ("healthy", "diseased")}
    print(f"Stage A healthy-vs-rest:   within AUC={auc['healthy'][0]:.3f}  external AUC={auc['healthy'][1]:.3f}")
    print(f"Stage B diseased-vs-rest:  within AUC={auc['diseased'][0]:.3f}  external AUC={auc['diseased'][1]:.3f}")

    # 4. Does z-scoring ALL seven features per flight help the external test? --
    print("\n" + "=" * 70 + "\n4. PER-FLIGHT STANDARDISATION OF ALL SEVEN FEATURES\n" + "=" * 70)
    z2 = ((A2[F7] - A2[F7].mean()) / A2[F7].std()).values
    z1 = ((A1[F7] - A1[F7].mean()) / A1[F7].std()).values
    z2s, z1s = fit_scale(z2, z1)
    Pz = combine(mk().fit(z2s, (y2 == "healthy").astype(int)).predict(z1s),
                 mk().fit(z2s, (y2 == "diseased").astype(int)).predict(z1s))
    f1_z = f1_score(y1, Pz, average="macro", labels=LO)
    dr_z = int((Pz[y1 == "diseased"] == "diseased").sum())
    print(f"External, all features z-scored per flight: macro-F1={f1_z:.3f}  diseased recall={dr_z}/{dn_e}")

    # 5. Bootstrap CI on external macro-F1 ------------------------------------
    print("\n" + "=" * 70 + f"\n5. BOOTSTRAP 95% CI, external macro-F1 ({N_BOOT} resamples)\n" + "=" * 70)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y1), len(y1))
        if len(set(y1[i])) < 2:
            continue
        boot.append(f1_score(y1[i], Pe[i], average="macro", labels=LO))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"External macro-F1 = {f1_e:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Save aggregate results (outputs/ is git-ignored) ------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"condition": "within_flight", "macro_f1": f1_w, "qwk": qwk_w, "diseased_recalled": dr_w, "diseased_total": dn_w,
         "stageA_auc": auc["healthy"][0], "stageB_auc": auc["diseased"][0]},
        {"condition": "external", "macro_f1": f1_e, "qwk": qwk_e, "diseased_recalled": dr_e, "diseased_total": dn_e,
         "stageA_auc": auc["healthy"][1], "stageB_auc": auc["diseased"][1], "ci_low": ci_lo, "ci_high": ci_hi},
    ]).to_csv(OUT_DIR / "locked_model_metrics.csv", index=False)

    # ------------------------------------------------------------------------
    # SANITY CHECK: every expected value was printed by cell 390 of my_code.html
    # Floats are compared at the 3 decimal places that were printed.
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70 + "\nSANITY CHECK against cell 390 of my_code.html\n" + "=" * 70)
    checks = [
        ("Within-flight macro-F1", float(round(f1_w, 3)), 0.438),
        ("Within-flight QWK", float(round(qwk_w, 3)), 0.206),
        ("Within-flight diseased recall", (dr_w, dn_w), (53, 86)),
        ("External macro-F1", float(round(f1_e, 3)), 0.394),
        ("External QWK", float(round(qwk_e, 3)), 0.206),
        ("External diseased recall", (dr_e, dn_e), (8, 25)),
        ("Matched plots", (len(matched), len(A1)), (523, 565)),
        ("Same label in both flights", (same, len(matched)), (427, 523)),
        ("Stage A AUC within / external", (float(round(auc["healthy"][0], 3)), float(round(auc["healthy"][1], 3))), (0.647, 0.665)),
        ("Stage B AUC within / external", (float(round(auc["diseased"][0], 3)), float(round(auc["diseased"][1], 3))), (0.716, 0.748)),
        ("All-feature per-flight z-score: macro-F1", float(round(f1_z, 3)), 0.308),
        ("All-feature per-flight z-score: diseased", dr_z, 16),
        ("Bootstrap 95% CI", (float(round(ci_lo, 3)), float(round(ci_hi, 3))), (0.344, 0.445)),
    ]
    all_ok = True
    for name, got, expected in checks:
        ok = got == expected
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name:42s} got {got}  expected {expected}")
    print("\nAll matched." if all_ok else "\nMISMATCH found. Do not trust any number above until it is explained.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
