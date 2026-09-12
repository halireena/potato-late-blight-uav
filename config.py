"""
config.py — the ONE place where file locations live.

Why this file exists
--------------------
The original notebook had lines like
    os.chdir("<absolute path to the thesis folder on my laptop>")
That path only works on one laptop, and it publishes your username and
folder layout. Every script should instead do:

    from config import DATA_DIR, PROCESSED_DIR, FIG_DIR, OUT_DIR, RANDOM_SEED

Anyone who has the data can point the code at it WITHOUT editing any code,
by setting an environment variable before running, e.g.

    export BLIGHT_DATA_DIR="/path/to/their/data"      (Mac / Linux)
    set BLIGHT_DATA_DIR=C:\\path\\to\\their\\data      (Windows)

If the variable is not set, the code looks in the repo's own data/ folder,
which is git-ignored, so the data never leaves your machine.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("BLIGHT_DATA_DIR", REPO_ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = Path(os.environ.get("BLIGHT_FIG_DIR", REPO_ROOT / "figures"))
OUT_DIR = Path(os.environ.get("BLIGHT_OUT_DIR", REPO_ROOT / "outputs"))

# Fixed seed used for every shuffle / split / model so reruns match.
RANDOM_SEED = 42

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def require(path: Path) -> Path:
    """Fail with a helpful message (instead of a confusing pandas error)
    when an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing input: {path}\n"
            "The trial data is not distributed with this repository.\n"
            "See data/README.md for the expected files, then set BLIGHT_DATA_DIR."
        )
    return path
