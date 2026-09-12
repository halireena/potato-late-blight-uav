"""
06_sensor_and_fusion.py
=======================
Two questions about what the imagery is worth, and whether DNA adds to it.

Run from the repo root, after 01_build_plot_table.py:
    python src/06_sensor_and_fusion.py

Question 1: is the expensive camera worth it?
---------------------------------------------
A multispectral camera sees red-edge and near-infrared, which a consumer RGB
camera cannot. It also costs many times more. So: split the features into the
ones an RGB drone could produce (canopy cover plus green and red texture) and
the ones only a multispectral sensor gives (red-edge and NIR indices and
texture), and run the locked cascade on each. Thirty different cross-validation
seeds, the same thirty for both, paired, so the comparison survives the luck of
any one split. The result is not the one the equipment budget expects.

Question 2: does adding the DNA help?
-------------------------------------
Each variety has a SNP fingerprint from the Sharma panel. Imagery says how a
plant looks now; genotype says what it inherited. Fusing them should help, if
the genetics carries blight signal at all. This tests it at the variety level,
where a genotype actually applies, on the subset of varieties that have both.

  SNP quality control  drop markers that are nearly monomorphic (MAF < 0.05)
                       or poorly genotyped (missing in > 20% of varieties),
                       then drop markers that ride along with a neighbour
                       (LD pruning, r-squared > 0.9 in a 50-marker window).
                       Both filters are label-free, so neither can leak.
  Early fusion         paste 5 genotype principal components onto the imagery
                       features and fit one model.
  Late fusion          fit imagery and genotype separately, average the two
                       predicted probabilities. Thirty seeds, paired.

A file this script deliberately does not read
---------------------------------------------
LD pruning walks each chromosome in physical order, so it needs marker
positions, which live in sharma_SNP_positions.xlsx. That file is present on
this machine but is under a standing instruction not to touch it, so it is NOT
linked into the data folder and the three statistics that depend on it are
skipped, not substituted:

    SNPs after LD pruning        the pruned count
    early fusion AUC             built on the pruned marker set
    late fusion, 30 seeds        built on the pruned marker set

The code for all three is written and sits below, gated only on the file being
present in the data folder. Linking it in is enough to make them run; nothing
here needs editing. Everything that does not need positions still runs.

Every expected value was printed by an executed cell of the thesis notebook
(mine_ML.ipynb, and 05_genomic_fusion.ipynb for the QC counts) and was supplied
as authorised. The script exits with code 1 on any mismatch.
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, OUT_DIR, PROCESSED_DIR, RANDOM_SEED, require  # noqa: E402

# See 04_permutation_tests.py for why this is silenced and nothing else is.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*converge.*")

TEXTURE_FULL_FILE = "texture_features_percentiles.csv"
GENOTYPE_FILE = "sharma_genotype_reduced_recoded_NA.csv"
SNP_POSITIONS_FILE = "sharma_SNP_positions.xlsx"

LOW_VIF = ["NDRE_mean", "canopy_cover_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
BANDS = ["Green", "Red", "RedEdge", "NIR"]
BAND_STATS = ["p10", "p90", "std", "min", "max"]
GENERIC = [f"{b}_{s}" for b in BANDS for s in BAND_STATS]
GENERIC_NORM = [c + "_norm" for c in GENERIC]
VARIETY_FEATURES = LOW_VIF + GENERIC_NORM

# What each sensor could actually deliver.
RGB_OBTAINABLE = (["canopy_cover_mean"]
                  + [f"Green_{s}_norm" for s in BAND_STATS]
                  + [f"Red_{s}_norm" for s in BAND_STATS])
MULTISPECTRAL_ONLY = (["NDRE_mean", "GNDVI_mean", "TCARI_OSAVI_mean"]
                      + [f"RedEdge_{s}_norm" for s in BAND_STATS]
                      + [f"NIR_{s}_norm" for s in BAND_STATS])

N_SEEDS = 30
MAF_MIN = 0.05
MAX_MISSING = 0.20
LD_WINDOW = 50
LD_R2 = 0.9
N_GENO_PCS = 5
PLOIDY = 4          # tetraploid: dosages run 0-4, so frequency = mean dosage / 4

KEEP_COLS = ["Disease_Plot_ID", "Variety_ID", "Disease_class"] + LOW_VIF


def lab(score):
    return "healthy" if score == 10 else ("moderate" if score >= 7 else "diseased")


def normalise_variety(name):
    return re.sub(r"[\s\-_()]", "", str(name)).upper()


def prep(Xtr, Xte, strategy="median"):
    im = SimpleImputer(strategy=strategy)
    Xtr, Xte = im.fit_transform(Xtr), im.transform(Xte)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def svc():
    return SVC(kernel="linear", class_weight="balanced", C=0.1)


def logreg():
    return LogisticRegression(class_weight="balanced", max_iter=5000)


def cascade(Xtr, ytr, Xte):
    a = svc().fit(Xtr, (ytr == "healthy").astype(int)).predict(Xte)
    b = svc().fit(Xtr, (ytr == "diseased").astype(int)).predict(Xte)
    return np.array(["healthy" if i == 1 else ("diseased" if j == 1 else "moderate")
                     for i, j in zip(a, b)])


def build_plot_table(flight, texture_full):
    """Processed plot table plus the twenty texture statistics, z-scored within
    this flight. Only the needed columns are carried over, because three of the
    generic texture names already exist in the processed table."""
    A = pd.read_csv(require(PROCESSED_DIR / f"A{flight}_clean_tex_norm.csv"))
    A["Disease_Plot_ID"] = A["Disease_Plot_ID"].astype(str)
    cols = [f"{b}_{flight}_{s}" for b in BANDS for s in BAND_STATS]
    t = texture_full[["Disease_Plot_ID"] + cols].copy()
    t.columns = ["Disease_Plot_ID"] + GENERIC
    out = A[KEEP_COLS].merge(t, on="Disease_Plot_ID", how="left")
    for c in GENERIC:
        out[c + "_norm"] = (out[c] - out[c].mean()) / out[c].std()
    out["y3"] = out["Disease_class"].apply(lab)
    out["vnorm"] = out["Variety_ID"].apply(normalise_variety)
    return out


def variety_rows(df):
    return df.groupby("vnorm").agg(
        **{f: (f, "mean") for f in VARIETY_FEATURES},
        mean_severity=("Disease_class", "mean"),
        n_plots=("Disease_class", "count"),
    ).reset_index()


def ld_prune(G, info, window=LD_WINDOW, r2_thresh=LD_R2):
    """Walk each chromosome in position order; drop a marker whose r-squared
    with a recently kept neighbour exceeds the threshold. Markers with no
    position are kept, because their LD cannot be judged."""
    col_of = {s: i for i, s in enumerate(info["snp"])}
    kept = []
    placed = info[info["chr"] != "unplaced"]
    for _, grp in placed.groupby("chr"):
        local = []
        for s in grp.sort_values("pos")["snp"]:
            c = col_of[s]
            keep = True
            for kc in local[-window:]:
                r = np.corrcoef(G[:, c], G[:, kc])[0, 1]
                if r * r > r2_thresh:
                    keep = False
                    break
            if keep:
                local.append(c)
        kept += local
    kept += [col_of[s] for s in info[info["chr"] == "unplaced"]["snp"]]
    return sorted(kept)


def variety_auc(X, y, extra=None, n_pcs=N_GENO_PCS):
    """Five-fold stratified CV at the variety level. Each row is one variety,
    so no grouping is needed: a variety cannot straddle a split.
    If `extra` is given, its principal components are pasted on (early fusion)."""
    T, P = [], []
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=RANDOM_SEED).split(X, y):
        a, b = prep(X[tr], X[te])
        if extra is not None:
            ga, gb = prep(extra[tr], extra[te], strategy="mean")
            pca = PCA(n_components=n_pcs, random_state=RANDOM_SEED).fit(ga)
            a = np.hstack([a, pca.transform(ga)])
            b = np.hstack([b, pca.transform(gb)])
        m = logreg().fit(a, y[tr])
        T.extend(y[te])
        P.extend(m.predict_proba(b)[:, 1])
    return roc_auc_score(T, P)


def main():
    texture_path = DATA_DIR / TEXTURE_FULL_FILE
    if not texture_path.exists():
        print(f"Missing input: {texture_path}\nNothing in this script can run without it.")
        sys.exit(1)
    texture_full = pd.read_csv(texture_path)
    texture_full["Disease_Plot_ID"] = texture_full["Disease_Plot_ID"].astype(str)

    A2 = build_plot_table("2", texture_full)
    checks, skipped, record = [], [], []

    # =====================================================================
    # 1. RGB-obtainable vs multispectral-only features
    # =====================================================================
    print("=" * 70)
    print("1. RGB-OBTAINABLE vs MULTISPECTRAL-ONLY FEATURES")
    print("=" * 70)
    y2, g2 = A2["y3"].values, A2["Variety_ID"].values

    def seed_scores(features):
        X = A2[features].values
        out = []
        for seed in range(N_SEEDS):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            T, P = [], []
            for tr, te in cv.split(X, y2, g2):
                a, b = prep(X[tr], X[te])
                P += list(cascade(a, y2[tr], b))
                T += list(y2[te])
            out.append(f1_score(T, P, average="macro"))
        return np.array(out)

    rgb = seed_scores(RGB_OBTAINABLE)
    ms = seed_scores(MULTISPECTRAL_ONLY)
    W_sensor, p_sensor = wilcoxon(ms, rgb)
    rgb_wins = int((rgb > ms).sum())
    print(f"   n = {len(A2)} plots, {N_SEEDS} paired cross-validation seeds")
    print(f"     RGB-obtainable   {len(RGB_OBTAINABLE):2d} features: macro-F1 {rgb.mean():.3f} +/- {rgb.std():.3f}")
    print(f"     Multispectral    {len(MULTISPECTRAL_ONLY):2d} features: macro-F1 {ms.mean():.3f} +/- {ms.std():.3f}")
    print(f"     RGB better in {rgb_wins}/{N_SEEDS} seeds")
    print(f"     Wilcoxon paired: W = {W_sensor:.1f}, p = {p_sensor:.4f}")
    print("     (the cheaper sensor wins; the red-edge and NIR bands did not pay for themselves)")
    checks.append(("RGB better than multispectral, seeds won", (rgb_wins, N_SEEDS), (24, 30)))
    checks.append(("RGB vs multispectral, Wilcoxon W", float(round(W_sensor, 1)), 56.0))
    checks.append(("RGB vs multispectral, Wilcoxon p", float(round(p_sensor, 4)), 0.0001))
    record.append({"statistic": "RGB wins over MS", "n": N_SEEDS, "value": rgb_wins})

    # =====================================================================
    # 2. Genomic panel and SNP quality control
    # =====================================================================
    print("\n" + "=" * 70)
    print("2. GENOMIC FUSION")
    print("=" * 70)
    geno_path = DATA_DIR / GENOTYPE_FILE
    if not geno_path.exists():
        why = f"{GENOTYPE_FILE} not found in {DATA_DIR}"
        print(f"   [SKIPPED] every genomic statistic: {why}")
        skipped.append(("All genomic statistics", why))
    else:
        geno = pd.read_csv(geno_path)
        geno["vnorm"] = geno["Variety.name"].apply(normalise_variety)
        snp_cols = [c for c in geno.columns if c not in ("Variety.name", "vnorm")]
        gmat = geno.set_index("vnorm")[snp_cols]
        G_all = gmat.values.astype(float)

        missingness = np.isnan(G_all).mean(axis=0)
        allele_freq = np.nanmean(G_all, axis=0) / PLOIDY
        maf = np.minimum(allele_freq, 1 - allele_freq)
        passes = (missingness <= MAX_MISSING) & (maf >= MAF_MIN)
        snps_qc = [snp_cols[i] for i in range(len(snp_cols)) if passes[i]]

        print(f"\n   SNP quality control   n = {len(geno)} varieties in the panel")
        print(f"     starting markers                    : {len(snp_cols)}")
        print(f"     fail MAF < {MAF_MIN}                      : {int((maf < MAF_MIN).sum())}")
        print(f"     fail missingness > {int(MAX_MISSING * 100)}%              : {int((missingness > MAX_MISSING).sum())}")
        print(f"     retained after MAF and missingness  : {len(snps_qc)}")
        checks.append(("Genomic panel, starting markers", len(snp_cols), 5718))
        checks.append(("Genomic panel, markers after MAF and missingness", len(snps_qc), 5041))
        record.append({"statistic": "SNPs after QC", "n": len(geno), "value": len(snps_qc)})

        # -- varieties present in both the trial and the panel -----------------
        V2 = variety_rows(A2)
        gset = set(geno["vnorm"])
        fusable = V2[V2["vnorm"].isin(gset)].set_index("vnorm")
        fusable["affected"] = (fusable["mean_severity"] < 10).astype(int)
        print(f"\n   Variety matching")
        print(f"     trial varieties (after merging spelling variants): {len(V2)}")
        print(f"     also in the genotype panel                       : {len(fusable)}")
        checks.append(("Varieties matched to the genotype panel", (len(fusable), len(V2)), (210, 282)))

        X_img = fusable[VARIETY_FEATURES].values
        y_fus = fusable["affected"].values
        auc_img = variety_auc(X_img, y_fus)
        print(f"\n   Imagery alone, variety level   n = {len(y_fus)} varieties,"
              f" {int(y_fus.sum())} affected")
        print(f"     AUC = {auc_img:.3f}")
        checks.append(("Imagery alone, 210-variety fusable subset, AUC",
                       float(round(auc_img, 3)), 0.615))
        record.append({"statistic": "imagery-alone variety AUC (fusable)",
                       "n": len(y_fus), "value": float(round(auc_img, 3))})

        # -- LD pruning and everything built on it ----------------------------
        pos_path = DATA_DIR / SNP_POSITIONS_FILE
        if not pos_path.exists():
            why = (f"{SNP_POSITIONS_FILE} is not in {DATA_DIR}. It exists on this machine but is "
                   "under a standing instruction not to touch it, so it was deliberately not "
                   "linked in. LD pruning needs marker positions and no other file carries them, "
                   "so nothing was substituted.")
            print(f"\n   [SKIPPED] LD pruning, early fusion and late fusion")
            print(f"     {SNP_POSITIONS_FILE} is not in the data folder.")
            print("     LD pruning walks each chromosome in position order, and only that file")
            print("     carries positions. No substitute was used. Link it in and these three")
            print("     statistics run with no code change.")
            skipped.append(("SNPs after LD pruning", why))
            skipped.append(("Early fusion AUC", why))
            skipped.append(("Late fusion, 30 seeds", why))
        else:
            pos = pd.read_excel(pos_path, sheet_name="annotation").set_index("solcap_SNP_ID")
            Gp = gmat[snps_qc].values.astype(float)
            colmean = np.nanmean(Gp, axis=0)
            nan_at = np.where(np.isnan(Gp))
            Gp[nan_at] = np.take(colmean, nan_at[1])   # label-free, only to get r-squared
            info = pd.DataFrame({
                "snp": snps_qc,
                "chr": [pos.loc[s, "chr_v403"] if s in pos.index else "unplaced" for s in snps_qc],
                "pos": [pos.loc[s, "position"] if s in pos.index else -1 for s in snps_qc],
            })
            kept_idx = ld_prune(Gp, info)
            snps_pruned = [snps_qc[i] for i in kept_idx]
            print(f"\n   LD pruning   r-squared > {LD_R2} in a {LD_WINDOW}-marker window")
            print(f"     {len(snps_qc)} -> {len(snps_pruned)} markers")
            checks.append(("Genomic panel, markers after LD pruning", len(snps_pruned), 4407))
            record.append({"statistic": "SNPs after LD pruning", "n": len(geno),
                           "value": len(snps_pruned)})

            G_fus = gmat.loc[fusable.index, snps_pruned].values.astype(float)
            auc_early = variety_auc(X_img, y_fus, extra=G_fus)
            print(f"\n   Early fusion, imagery + {N_GENO_PCS} genotype PCs   n = {len(y_fus)} varieties")
            print(f"     AUC = {auc_early:.3f}   (imagery alone {auc_img:.3f})")
            checks.append(("Early fusion, variety-level AUC", float(round(auc_early, 3)), 0.617))
            record.append({"statistic": "early fusion variety AUC", "n": len(y_fus),
                           "value": float(round(auc_early, 3))})

            img_seeds, late_seeds = [], []
            for seed in range(N_SEEDS):
                Ti, Pi, Tl, Pl = [], [], [], []
                for tr, te in StratifiedKFold(n_splits=5, shuffle=True,
                                              random_state=seed).split(X_img, y_fus):
                    a, b = prep(X_img[tr], X_img[te])
                    ga, gb = prep(G_fus[tr], G_fus[te], strategy="mean")
                    pca = PCA(n_components=N_GENO_PCS, random_state=RANDOM_SEED).fit(ga)
                    prob_img = logreg().fit(a, y_fus[tr]).predict_proba(b)[:, 1]
                    prob_gen = logreg().fit(pca.transform(ga), y_fus[tr]).predict_proba(
                        pca.transform(gb))[:, 1]
                    Ti.extend(y_fus[te]); Pi.extend(prob_img)
                    Tl.extend(y_fus[te]); Pl.extend((prob_img + prob_gen) / 2)
                img_seeds.append(roc_auc_score(Ti, Pi))
                late_seeds.append(roc_auc_score(Tl, Pl))
            img_seeds, late_seeds = np.array(img_seeds), np.array(late_seeds)
            W_fus, p_fus = wilcoxon(late_seeds, img_seeds)
            img_wins = int((img_seeds > late_seeds).sum())
            print(f"\n   Late fusion, averaged probabilities   n = {len(y_fus)} varieties,"
                  f" {N_SEEDS} paired seeds")
            print(f"     imagery alone {img_seeds.mean():.3f} +/- {img_seeds.std():.3f}"
                  f" | late fusion {late_seeds.mean():.3f} +/- {late_seeds.std():.3f}")
            print(f"     imagery better in {img_wins}/{N_SEEDS} seeds")
            print(f"     Wilcoxon paired: W = {W_fus:.1f}, p = {p_fus:.4f}")
            print("     (adding the DNA makes it reliably worse, not better)")
            checks.append(("Late fusion loses to imagery, seeds", (img_wins, N_SEEDS), (23, 30)))
            checks.append(("Late fusion, Wilcoxon W", float(round(W_fus, 1)), 101.0))
            checks.append(("Late fusion, Wilcoxon p", float(round(p_fus, 4)), 0.0058))
            record.append({"statistic": "imagery wins over late fusion", "n": N_SEEDS,
                           "value": img_wins})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(record).to_csv(OUT_DIR / "sensor_and_fusion.csv", index=False)

    # =====================================================================
    print("\n" + "=" * 70)
    print("SANITY CHECK against authorised values from the thesis notebooks")
    print("=" * 70)
    all_ok = True
    for name, got, expected in checks:
        ok = got == expected
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name:52s} got {got}  expected {expected}")
    if skipped:
        print()
        seen = set()
        for name, why in skipped:
            print(f"  [SKIPPED] {name}")
            if why not in seen:
                print(f"            {why}")
                seen.add(why)
        print(f"\n  {len(skipped)} statistic(s) skipped. The run is INCOMPLETE:")
        print("  a skipped statistic is neither confirmed nor refuted.")
    print("\nAll matched." if all_ok
          else "\nMISMATCH found. Do not trust any number above until it is explained.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
