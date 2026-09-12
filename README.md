# <!-- TODO: your title --> Potato late blight severity from UAV imagery with machine learning

<!-- TODO: 2–3 sentences in your own words: what question the code answers, what the inputs are, what comes out. -->

MSc Bioinformatics thesis code, University of Birmingham Dubai. <!-- TODO: year -->

> **The data is not included in this repository.**
> This repo contains analysis code only. The UAV imagery, plot-level disease scores and variety information
> were collected in a field trial and are not publicly available. See [Data availability](#data-availability).

---

## Repository layout

```
.
├── reproduce.py           # ONE command: runs everything and says REPRODUCED or NOT REPRODUCED
├── config.py              # all file locations + the fixed random seed (edit nothing else to relocate data)
├── requirements.txt       # exact package versions used for the thesis results
├── src/                   # analysis scripts, run in numbered order
├── tools/
│   └── check_for_leaks.py # run before every push
├── data/                  # EMPTY in the repo; your local copy of the data goes here (git-ignored)
├── figures/               # generated locally, git-ignored
└── outputs/               # generated locally, git-ignored
```

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `src/01_build_plot_table.py` | joins zonal statistics, disease scores, varieties and texture into one table per flight | `data/` | `data/processed/` |
| `src/02_locked_model.py` | two-stage linear SVM cascade: grouped cross-validation on the training flight, then the temporally independent test flight | `data/processed/` | `outputs/` |
| `src/03_architectures.py` | the rejected comparator architectures, each run within-flight and on the test flight, against the majority-class floor | `data/processed/` | `outputs/` |
| `src/04_permutation_tests.py` | permutation tests: recomputes each headline result 1,000 times on shuffled labels to see what noise alone produces (slow, ~15 min) | `data/processed/`, `data/` | `outputs/` |

Each script ends with a sanity check that compares its results with the values printed in the original
analysis notebook, and exits with an error on any mismatch.

<!-- TODO: add later scripts here as they are added (permutation tests, sensor comparison, variety-level, fusion, figures). -->

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Results were produced with Python 3.9.6 on macOS (arm64). The pinned `requirements.txt` installs on Python 3.9.

## Running the code (requires access to the data)

1. Place the input files described in [`data/README.md`](data/README.md) in `data/`,
   or point the code at another folder:
   ```bash
   export BLIGHT_DATA_DIR="/path/to/data"
   ```
2. Run the whole pipeline and check it against the thesis in one command:
   ```bash
   python reproduce.py
   ```
   It checks the environment against `requirements.txt`, checks the input files, runs every script in `src/`
   in order, and ends with **REPRODUCED** or **NOT REPRODUCED**. A dated report, stamped with the git commit,
   is written to `outputs/reproducibility_report.txt`.

   The scripts can also be run one at a time: `python src/01_build_plot_table.py`, then `python src/02_locked_model.py`.

Without the data, the scripts stop with a message explaining which file is missing.

## Methods in brief

<!-- TODO: write this yourself, short. Points the code implements that a reader needs to know:
- Two UAV flights; which flight trains the model and which is held out as a temporally independent test.
- Labels: 1–10 visual severity score mapped to three classes (10 = healthy, 7–9 = moderate, 1–6 = diseased).
- Leakage control: StratifiedGroupKFold (5 folds) grouped by variety, so replicate plots of a variety never
  sit on both sides of a split; imputation and scaling are fitted inside each training fold only.
- Final model and features: confirm against your locked results before writing.
-->

## Reproducibility

- A single random seed (`RANDOM_SEED` in `config.py`) is used for every split and model.
- Exact package versions are listed in `requirements.txt`.

## Data availability

<!-- TODO: agree this wording with your supervisor / the data owner before the repo goes public. -->
The field trial data used in this study are not publicly available. <!-- TODO: access conditions, e.g.
"Requests for access can be directed to [data owner / institution]." -->

Genotype data were derived from Sharma et al. (2018), *G3: Genes|Genomes|Genetics*; obtain them from the
original publication rather than from this repository.

## Citation

See `CITATION.cff` (GitHub shows it as "Cite this repository"). <!-- TODO: add the Zenodo DOI badge after the first release. -->

## Licence

<!-- TODO: choose one when creating the repo (MIT is common for code). -->

## Contact

Halireena Rushdiha Mohomed <!-- TODO: email or LinkedIn -->
