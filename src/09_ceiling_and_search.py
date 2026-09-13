"""
09_ceiling_and_search.py
========================
How high could this have gone? The analyses behind Results 2.9, Figure 8 and
Supplementary Table S5.

Run from the repo root, after 01_build_plot_table.py:
    python src/09_ceiling_and_search.py

The idea in plain words
-----------------------
Everything else in src/ asks how well the locked model does. This asks whether
anything else could have done better, and answers it by trying: recalibrating
the decision thresholds, searching 102 model configurations properly, and
shrinking the training classes to see whether more data would have helped.

    1. Quantile matching     the test flight's decision scores are shifted onto
                             the training flight's score distribution. Uses NO
                             test labels, so it is the only recalibration that
                             could honestly be deployed.
    2. Oracle grid search    both stage thresholds tuned directly against the
                             test labels. Not deployable -- it reads the answer
                             key -- so it is an upper bound, not a result.
    3. Nested search         102 combinations of feature set, architecture and
                             hyperparameters, five outer by five inner folds,
                             grouped by variety so a variety never straddles a
                             split. The outer folds never inform the choice, so
                             the outer mean is an honest estimate of what
                             searching this space buys.
    4. Moderate curve        the moderate class subsampled from 10 plots to 47,
                             scored with the locked cascade on the seven features.
    5. Diseased curve        the diseased class subsampled from 10 plots to 86, scored
                             with one balanced logistic regression on the low-VIF four.
                             The two curves are different experiments, not one method
                             run twice, and they do not answer the same question.
    6. Drift ratios          each index's training-flight mean over its
                             test-flight mean: how far the sensor moved between
                             the two dates.

Two analyses from this section are NOT here
-------------------------------------------
They are skipped rather than guessed at, and the run says so:

    Stage B threshold sweep and cost-sensitive training (1:1 to 20:1).
        The recorded result is that neither beats the locked test-flight
        diseased recall of 0.320. Whether that holds depends entirely on where
        the threshold and the cost ratio were chosen -- on the training flight,
        on out-of-fold scores, or on the test flight itself -- and those give
        different answers. I could not find the cell that fixes it, and a
        reconstruction that happened to satisfy the check would be worthless.

    Six moderate-class transformations, all AUCs between 0.457 and 0.542.
        Four are traceable: raw normalised features (0.457), sigmoid-calibrated
        (0.542), per-flight rank transform (0.494) and LDA (0.459). The other
        two are not, and the check is on all six, so checking four would not be
        the recorded claim.

Both need one thing each: the notebook cell that defines them. Point me at
those and they drop straight in.

Every expected value here was printed by an executed cell of the thesis
notebook (mine_ML.ipynb) and was supplied as authorised. Two findings are
printed WITHOUT a check, because they are observations rather than recorded
values: whether the learning curves are flat, and the two different moderate
curves on record. The script exits with code 1 on any mismatch.

The nested search fits 2,550 models. Expect this script to take a while.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*converge.*")

LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
TEX_N = ["NIR_max_norm", "RedEdge_p90_norm", "Red_min_norm"]
F7 = LOW_VIF + TEX_N
BASE_6 = ["canopy_cover_mean", "NDVI_mean", "NDRE_mean",
          "GNDVI_mean", "EVI2_mean", "OSAVI_mean"]
NINE = ["canopy_cover_mean", "NDVI_mean", "RDVI_mean", "GNDVI_mean", "NDRE_mean",
        "PVR_mean", "TVI_mean", "TCARI_mean", "TCARI_OSAVI_mean"]
ALL_INDICES = ["PVR", "canopy_cover", "NDVI", "GNDVI", "NDRE", "TCARI_OSAVI",
               "TCARI", "RDVI", "OSAVI", "EVI2", "TVI"]
LO = ["diseased", "moderate", "healthy"]
INNER_SEED = 1        # the nested search seeds its INNER folds 1, not RANDOM_SEED

FEATURE_SETS = {"LowVIF_4": LOW_VIF, "Baseline_6": BASE_6, "NoEVI2OSAVI_9": NINE}
MODEL_GRIDS = {
    "LogReg": {"model": LogisticRegression,
               "fixed": {"class_weight": "balanced", "max_iter": 5000},
               "grid": {"C": [0.01, 0.1, 1.0, 10.0]}},
    "RandomForest": {"model": RandomForestClassifier,
                     "fixed": {"class_weight": "balanced", "random_state": RANDOM_SEED},
                     "grid": {"n_estimators": [100, 300, 500],
                              "max_depth": [3, 5, 10, None]}},
    "XGBoost": {"model": XGBClassifier,
                "fixed": {"eval_metric": "mlogloss", "random_state": RANDOM_SEED,
                          "verbosity": 0},
                "grid": {"n_estimators": [100, 300], "max_depth": [3, 5, 7],
                         "learning_rate": [0.01, 0.1, 0.3]}},
}


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def prep(Xtr, Xte):
    im = SimpleImputer(strategy="median")
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def svc():
    return SVC(kernel="linear", class_weight="balanced", C=0.1)


def grouped_cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def cascade_pred(score_A, score_B, thr_A=0.0, thr_B=0.0):
    """The cascade decision rule, read at arbitrary thresholds."""
    return np.array(["healthy" if a else ("diseased" if b else "moderate")
                     for a, b in zip(score_A > thr_A, score_B > thr_B)])


def quantile_match(test_scores, ref_scores):
    """Map test scores onto the reference distribution by rank. Uses only the
    two score distributions, never the test labels."""
    ranks = rankdata(test_scores, method="average") / (len(test_scores) + 1)
    return np.quantile(ref_scores, ranks)


def diseased_recall(y_true, y_pred):
    m = y_true == "diseased"
    return float((y_pred[m] == "diseased").sum() / m.sum())


def main():
    A2 = pd.read_csv(require(PROCESSED_DIR / "A2_clean_tex_norm.csv"))
    A1 = pd.read_csv(require(PROCESSED_DIR / "A1_clean_tex_norm.csv"))
    for d in (A2, A1):
        d["y3"] = d["Disease_class"].apply(lab)
    y2, y1 = A2["y3"].values, A1["y3"].values
    g2 = A2["Variety_ID"].values
    checks, skipped = [], []

    print("=" * 74)
    print("CEILING ANALYSES AND THE ARCHITECTURE SEARCH")
    print("=" * 74)

    # =====================================================================
    # 1 and 2. Threshold recalibration: deployable, then oracle
    # =====================================================================
    print("\n1. THRESHOLD RECALIBRATION OF THE LOCKED CASCADE")
    X2s, X1s = prep(A2[F7].values, A1[F7].values)
    mA = svc().fit(X2s, (y2 == "healthy").astype(int))
    mB = svc().fit(X2s, (y2 == "diseased").astype(int))
    sA_test, sB_test = mA.decision_function(X1s), mB.decision_function(X1s)

    locked_pred = cascade_pred(sA_test, sB_test)
    locked_f1 = f1_score(y1, locked_pred, average="macro", labels=LO)
    locked_recall = diseased_recall(y1, locked_pred)

    # out-of-fold training scores: the reference distribution
    oof_A, oof_B = [], []
    for tr, te in grouped_cv().split(A2[F7].values, y2, g2):
        a, b = prep(A2[F7].values[tr], A2[F7].values[te])
        oof_A.extend(svc().fit(a, (y2[tr] == "healthy").astype(int)).decision_function(b))
        oof_B.extend(svc().fit(a, (y2[tr] == "diseased").astype(int)).decision_function(b))
    oof_A, oof_B = np.array(oof_A), np.array(oof_B)

    q_pred = cascade_pred(quantile_match(sA_test, oof_A), quantile_match(sB_test, oof_B))
    q_f1 = float(round(f1_score(y1, q_pred, average="macro", labels=LO), 3))
    q_recall = float(round(diseased_recall(y1, q_pred), 3))
    print(f"   n = {len(y1)} test plots, {int((y1 == 'diseased').sum())} diseased")
    print(f"     locked, thresholds at 0     : macro-F1 {locked_f1:.3f}, diseased recall {locked_recall:.3f}")
    print(f"     quantile-matched (deployable): macro-F1 {q_f1:.3f}, diseased recall {q_recall:.3f}")
    print("     (uses no test labels; recall roughly doubles, macro-F1 falls below the floor)")
    checks.append(("Quantile-matched, diseased recall", q_recall, 0.640))
    checks.append(("Quantile-matched, macro-F1", q_f1, 0.297))

    grid = np.linspace(-2, 2, 41)
    best_f1, best_thr = -1.0, None
    for thr_A in grid:
        for thr_B in grid:
            f1v = f1_score(y1, cascade_pred(sA_test, sB_test, thr_A, thr_B),
                           average="macro", labels=LO)
            if f1v > best_f1:
                best_f1, best_thr = f1v, (thr_A, thr_B)
    oracle_f1 = float(round(best_f1, 3))
    print(f"\n2. ORACLE CEILING, both thresholds tuned on the test labels")
    print(f"   grid = {len(grid)} x {len(grid)} threshold pairs")
    print(f"     best macro-F1 {oracle_f1:.3f} at thresholds A={best_thr[0]:.2f}, B={best_thr[1]:.2f}")
    print("     NOT deployable: it reads the answer key. An upper bound, not a result.")
    checks.append(("Oracle threshold ceiling, macro-F1", oracle_f1, 0.441))

    # =====================================================================
    # SKIPPED: threshold sweep and cost-sensitive training
    # =====================================================================
    why = ("the cell fixing where the threshold and the cost ratio are chosen could not be "
           "found; on the test flight the oracle above already beats 0.320, so the recorded "
           "claim depends on a protocol that is not written down")
    print("\n   [SKIPPED] Stage B threshold sweep and cost-sensitive training (1:1 to 20:1)")
    print(f"     {why}")
    skipped.append(("Threshold sweep and cost-sensitive training", why))

    # =====================================================================
    # 3. Nested search over 102 combinations
    # =====================================================================
    n_combos = sum(len(list(ParameterGrid(s["grid"]))) for s in MODEL_GRIDS.values()) * len(FEATURE_SETS)
    print(f"\n3. NESTED SEARCH, {n_combos} combinations, 5 outer x 5 inner folds")
    le = LabelEncoder().fit(y2)
    y_enc = le.transform(y2)
    outer_scores = []
    for fold, (tr, te) in enumerate(grouped_cv().split(A2[LOW_VIF].values, y_enc, g2), 1):
        g_tr = g2[tr]
        best_inner, best = -1.0, None
        for fname, fcols in FEATURE_SETS.items():
            Xf = A2[fcols].values
            for mname, spec in MODEL_GRIDS.items():
                for params in ParameterGrid(spec["grid"]):
                    inner = []
                    # The inner splitter is seeded 1, not RANDOM_SEED. That is what
                    # the notebook does, and the winners it picks depend on it.
                    for itr, ite in StratifiedGroupKFold(
                            n_splits=5, shuffle=True, random_state=INNER_SEED).split(
                            Xf[tr], y_enc[tr], g_tr):
                        a, b = prep(Xf[tr][itr], Xf[tr][ite])
                        m = spec["model"](**spec["fixed"], **params).fit(a, y_enc[tr][itr])
                        inner.append(f1_score(y_enc[tr][ite], m.predict(b), average="macro"))
                    score = float(np.mean(inner))
                    if score > best_inner:
                        best_inner, best = score, (fname, fcols, mname, spec, params)
        fname, fcols, mname, spec, params = best
        Xf = A2[fcols].values
        a, b = prep(Xf[tr], Xf[te])
        m = spec["model"](**spec["fixed"], **params).fit(a, y_enc[tr])
        outer = f1_score(y_enc[te], m.predict(b), average="macro")
        outer_scores.append(outer)
        print(f"   outer fold {fold}: winner {fname} / {mname} {params}"
              f" -> inner {best_inner:.3f}, honest outer {outer:.3f}", flush=True)
    outer_scores = np.array(outer_scores)
    mean_outer = float(round(outer_scores.mean(), 3))
    sd_outer = float(round(outer_scores.std(ddof=1), 3))
    print(f"   outer-fold mean {mean_outer:.3f}, SD {sd_outer:.3f}  (n = 5 outer folds)")
    checks.append(("Nested search, outer-fold mean", mean_outer, 0.361))
    checks.append(("Nested search, outer-fold SD", sd_outer, 0.043))
    checks.append(("Nested search, per-fold scores",
                   [float(round(v, 3)) for v in outer_scores],
                   [0.320, 0.430, 0.369, 0.329, 0.358]))

    # =====================================================================
    # SKIPPED: six moderate-class transformations
    # =====================================================================
    why2 = ("only four of the six transformations are traceable (raw 0.457, sigmoid-calibrated "
            "0.542, per-flight rank 0.494, LDA 0.459); the recorded check is on all six")
    print("\n   [SKIPPED] Six moderate-class transformations")
    print(f"     {why2}")
    skipped.append(("Six moderate-class transformations", why2))

    # =====================================================================
    # 4 and 5. Learning curves
    # =====================================================================
    # The two curves are NOT the same experiment, and mixing them up gives the
    # wrong answer. The moderate curve runs the locked SVM cascade on the seven
    # locked features and averages one score per repeat; the diseased curve runs
    # a single balanced LogisticRegression on the low-VIF four and averages over
    # every fold of every repeat. Each is reproduced as it was actually run.
    X7, gall = A2[F7].values, A2["Variety_ID"].values

    def curve_cascade(target, sizes, n_repeats=10):
        """Moderate-class curve: locked cascade, seven features. One shared RNG
        walks across every size and repeat, as in the notebook."""
        rows = []
        rng = np.random.RandomState(RANDOM_SEED)
        hit_idx = np.where(y2 == target)[0]
        other_idx = np.where(y2 != target)[0]
        for n in sizes:
            f1s, recs = [], []
            for _ in range(n_repeats):
                keep = np.concatenate([other_idx, rng.choice(hit_idx, n, replace=False)])
                Xs, ys, gs = X7[keep], y2[keep], gall[keep]
                T, P = [], []
                for tr, te in StratifiedGroupKFold(
                        n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(Xs, ys, gs):
                    a, b = prep(Xs[tr], Xs[te])
                    ytr = ys[tr]
                    pa = svc().fit(a, (ytr == "healthy").astype(int)).predict(b)
                    pb = svc().fit(a, (ytr == "diseased").astype(int)).predict(b)
                    P += ["healthy" if i == 1 else ("diseased" if j == 1 else "moderate")
                          for i, j in zip(pa, pb)]
                    T += list(ys[te])
                T, P = np.array(T), np.array(P)
                f1s.append(f1_score(T, P, average="macro"))
                m = T == target
                recs.append(float((P[m] == target).sum() / m.sum()) if m.sum() else 0.0)
            rows.append({"n": n, "f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                         "recall": float(np.mean(recs)), "recall_sd": float(np.std(recs))})
        return pd.DataFrame(rows)

    def curve_logreg(target, sizes, n_repeats=10):
        """Diseased-class curve: one balanced LogisticRegression on the low-VIF
        four, scored per fold, subsample seeded by the repeat number."""
        rows = []
        mask = A2["y3"] == target
        pool = A2[mask].reset_index(drop=True)
        rest = A2[~mask].reset_index(drop=True)
        for n in sizes:
            f1s, recs = [], []
            for repeat in range(n_repeats):
                sub = pd.concat([rest, pool.sample(n=n, random_state=repeat)], ignore_index=True)
                X, y, g = sub[LOW_VIF].values, sub["y3"].values, sub["Variety_ID"].values
                for tr, te in StratifiedGroupKFold(
                        n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, y, g):
                    a, b = prep(X[tr], X[te])
                    pred = LogisticRegression(class_weight="balanced",
                                              max_iter=5000).fit(a, y[tr]).predict(b)
                    f1s.append(f1_score(y[te], pred, average="macro"))
                    hit = y[te] == target
                    if hit.sum():
                        recs.append(float((pred[hit] == target).mean()))
            rows.append({"n": n, "f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                         "recall": float(np.mean(recs)), "recall_sd": float(np.std(recs))})
        return pd.DataFrame(rows)

    # The last size is the whole class, read from the data rather than typed in,
    # so no plot count is hard-coded here.
    n_moderate = int((A2["y3"] == "moderate").sum())
    n_diseased = int((A2["y3"] == "diseased").sum())

    print("\n4. MODERATE-CLASS LEARNING CURVE")
    mod = curve_cascade("moderate", [10, 15, 20, 25, 30, 35, 40] + [n_moderate])
    for _, r in mod.iterrows():
        print(f"   n={int(r['n']):3d}: macro-F1 {r['f1']:.3f} +/- {r['f1_sd']:.3f}"
              f" | moderate recall {r['recall']:.3f} +/- {r['recall_sd']:.3f}")
    checks.append(("Moderate curve, recall at n=10", float(round(mod.iloc[0]["recall"], 3)), 0.020))
    checks.append(("Moderate curve, recall at n=47", float(round(mod.iloc[-1]["recall"], 3)), 0.196))

    print("\n5. DISEASED-CLASS LEARNING CURVE")
    dis = curve_logreg("diseased", [10, 20, 30, 40, 50, 60, 70] + [n_diseased])
    for _, r in dis.iterrows():
        print(f"   n={int(r['n']):3d}: macro-F1 {r['f1']:.3f} +/- {r['f1_sd']:.3f}"
              f" | diseased recall {r['recall']:.3f} +/- {r['recall_sd']:.3f}")
    checks.append(("Diseased curve, macro-F1 at n=10", float(round(dis.iloc[0]["f1"], 3)), 0.268))
    checks.append(("Diseased curve, macro-F1 at n=86", float(round(dis.iloc[-1]["f1"], 3)), 0.369))

    # =====================================================================
    # 6. Inter-flight drift
    # =====================================================================
    print("\n6. INTER-FLIGHT DRIFT RATIOS, training-flight mean over test-flight mean")
    drift = {i: A2[f"{i}_mean"].mean() / A1[f"{i}_mean"].mean() for i in ALL_INDICES}
    print(f"   n = {len(A2)} training plots, {len(A1)} test plots")
    for i, v in sorted(drift.items(), key=lambda kv: kv[1]):
        flag = "" if 0.5 <= v <= 2 else "   <- moved by more than a factor of two"
        print(f"     {i:16s} {v:6.2f}{flag}")
    for name, expected in (("TVI", 9.0), ("EVI2", 6.1), ("OSAVI", 3.1), ("RDVI", 2.7)):
        checks.append((f"Drift ratio, {name}", float(round(drift[name], 1)), expected))

    # =====================================================================
    # PRINTED, NOT CHECKED
    # =====================================================================
    print("\n" + "=" * 74)
    print("PRINTED BUT NOT CHECKED")
    print("=" * 74)

    def is_monotonic(series):
        vals = list(series)
        return all(b >= a for a, b in zip(vals, vals[1:]))

    print("\n  Are the learning curves flat?")
    for name, df, col in (("moderate", mod, "f1"), ("diseased", dis, "f1")):
        vals = df[col].round(3).tolist()
        rng = max(vals) - min(vals)
        mono = is_monotonic(df[col])
        print(f"    {name:9s} macro-F1 {vals[0]:.3f} -> {vals[-1]:.3f}, spread {rng:.3f}")
        print(f"              rises monotonically: {'YES' if mono else 'no'}")
        print(f"              full curve: {vals}")
        verdict = ("NOT flat. It rises at every step." if mono
                   else ("NOT flat: it rises overall." if vals[-1] > vals[0] + 0.02
                         else "flat within noise."))
        print(f"              PLAIN ANSWER: {verdict}")
    print("\n    Both curves rise, though only the diseased one rises at every single step;")
    print("    the moderate one dips at two points while climbing overall.")
    print("    If the Discussion says the curves are flat, and uses that")
    print("    to argue the ceiling is a signal problem rather than a sample-size problem,")
    print("    that argument does not follow from these curves. More plots of the rare")
    print("    classes did keep helping, all the way to the largest size available.")

    print("\n  Two moderate-class curves are on record, and they disagree:")
    print("    earlier run, macro-F1 0.347 -> 0.369, described as flat")
    print("      from the comparison note printed by cell 380, the diseased-curve cell")
    print("    later run,   macro-F1 0.395 -> 0.440")
    print("      from cell 504 (execution 1869), PART 1, the moderate learning curve")
    print(f"    this run,    macro-F1 {mod.iloc[0]['f1']:.3f} -> {mod.iloc[-1]['f1']:.3f}")
    print("    The two recorded runs use different models. Cell 504 runs the locked SVM")
    print("    cascade on the seven locked features; that is what the moderate curve above")
    print("    reproduces. The earlier 0.347-to-0.369 figure comes from the other method,")
    print("    a balanced logistic regression on the low-VIF four, which is what the")
    print("    diseased curve above uses. Different models, different curves, and the")
    print("    'flat' description belongs to the earlier one. Neither is checked here.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mod.assign(curve="moderate").to_csv(OUT_DIR / "learning_curve_moderate.csv", index=False)
    dis.assign(curve="diseased").to_csv(OUT_DIR / "learning_curve_diseased.csv", index=False)
    pd.Series(drift).to_csv(OUT_DIR / "drift_ratios.csv")

    # =====================================================================
    print("\n" + "=" * 74)
    print("SANITY CHECK against authorised values from the thesis notebook")
    print("=" * 74)
    all_ok = True
    for name, got, expected in checks:
        ok = got == expected
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name:44s} got {got}  expected {expected}")
    if skipped:
        print()
        for name, reason in skipped:
            print(f"  [SKIPPED] {name}")
        print(f"\n  {len(skipped)} analysis/analyses skipped rather than guessed at. The run is")
        print("  INCOMPLETE: a skipped analysis is neither confirmed nor refuted.")
    print("\nAll matched." if all_ok
          else "\nMISMATCH found. Do not trust any number above until it is explained.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
