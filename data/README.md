# data/ — expected inputs (not distributed)

Nothing in this folder is uploaded to GitHub except this file. It describes the **structure** of the inputs
so the code can be understood and rerun by someone with authorised access. No values are listed here.

## Inputs read by `src/01_build_plot_table.py`

**Zonal statistics per flight**: `ALL_stats_1_JB_jack_data.csv`, `ALL_stats_2_JB_jack_data.csv`.
Several rows per plot. Key column `Disease_Plot_ID` (grid rows labelled `G` are dropped).
Feature columns follow `<index>_<flight>_<stat>`, where index is one of
canopy_cover, NDVI, RDVI, GNDVI, NDRE, PVR, EVI2, TVI, TCARI, OSAVI, TCARI_OSAVI and stat is mean, median or stdev
(canopy_cover has no median).

**Disease scores**: `final_disease_data.csv`.
Columns `Disease_Plot_ID`, `Flight` (`Flight 1` or `Flight 2`), `Disease_class` (integer visual score, 1 to 10).

**Plot-to-variety lookup**: `variety_and_plot_IDs.xlsx`.
Sheet `2024 Conv Block Labels` (header on the second row) with columns `Variety` and `2024 Plot No`;
sheet `2024 Map` (no header) holding the sub-trial variety and its two replicate plot numbers in the
34th to 36th columns.

**Variety name corrections**: `variety_spelling_fixes.csv`.
Two columns, `wrong` and `right`. Merges spelling variants of the same variety so its replicate plots
share one cross-validation group. Kept here rather than in the code so no variety names are published.

**Texture statistics per plot**: `texture_features_selected.csv`.
Columns `Disease_Plot_ID`, `NIR_<flight>_max`, `RedEdge_<flight>_p90`, `Red_<flight>_min` for flights 1 and 2.

## Created by the pipeline

`processed/A2_clean_tex_norm.csv` (Flight 2, training) and `processed/A1_clean_tex_norm.csv` (Flight 1, test):
one row per scored plot, with `Disease_Plot_ID`, `Variety_ID`, `Disease_class`, the zonal statistics,
the raw texture columns and their per-flight z-scores (`*_norm`).
