#!/usr/bin/env python3
"""
check_for_leaks.py
==================
Run this BEFORE every push, and definitely before switching the repo to public.

    python tools/check_for_leaks.py                 # scan the whole repo
    python tools/check_for_leaks.py --self-test     # prove the scanner itself works
    python tools/check_for_leaks.py src/model.py    # scan specific files

Optional, strongest check (catches variety names typed into the code):
    python tools/check_for_leaks.py --terms-from-table "/path/OUTSIDE/repo/variety_and_plot_IDs.xlsx" --column Variety

Two levels of result
--------------------
  BLOCK   Must be fixed before pushing. The script exits with code 1.
  REVIEW  Look at it yourself. It may be fine (for example, a confusion matrix
          that is already printed in the thesis), or it may be raw data typed
          into the code by hand.

What it looks for, and why each one matters
-------------------------------------------
  1. Data / figure / notebook files that git would upload.
     .gitignore should stop these, but a typo in .gitignore fails silently.
  2. Data files that were EVER committed, even if deleted later.
     Deleting a file does not remove it from git history; anyone can check out
     an old commit and get it back.
  3. Notebooks (.ipynb) with saved outputs.
     A single df.head() output cell publishes real rows of the dataset.
  4. Files over 1 MB. Code is small; big files are usually data.
  5. Personal absolute paths (for example a Mac home folder path).
     They reveal your username and folder layout, and they break the code for
     everyone else.
  6. Private terms such as variety names, loaded from a file kept OUTSIDE the
     repo. If a name from the trial appears in the code, raw data has been
     typed in by hand.
     Two deliberate exemptions, both narrow, so that the check stays usable:
       - Your own name, as published in CITATION.cff, does not count as your
         username leaking. See PUBLISHED NAME below.
       - A private term that is also an ordinary word (ALPHA, EDEN) is checked
         case-sensitively instead of being dropped. See COMMON_WORD_TERMS.
  7. Hard-coded values that look like data: per-plot score lists, long lists of
     numbers, integer matrices, quoted plot IDs.
"""
from __future__ import annotations

import argparse
import getpass
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_EXT = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather", ".pkl", ".pickle",
    ".joblib", ".npy", ".npz", ".h5", ".hdf5", ".json", ".sqlite", ".db",
    ".tif", ".tiff", ".gpkg", ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qgz", ".qgs",
}
FIGURE_EXT = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}
TEXT_EXT = {".py", ".r", ".md", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".cff", ".ini", ".sh", ".rmd"}
MAX_BYTES = 1_000_000
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints", "node_modules"}

PATH_PATTERNS = [
    re.compile(r"/Users/[^/\s'\"]+"),                      # macOS home folders
    re.compile(r"/home/[^/\s'\"]+"),                       # Linux home folders
    re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\s'\"]+"),     # Windows home folders
    re.compile(r"~/(Desktop|Downloads|Documents|OneDrive)"),
]
LITERAL_PATTERNS = [
    ("per-plot score list typed into the code", re.compile(r"['\"]scores?['\"]\s*:\s*\[")),
    ("long list of numbers (8+) typed into the code",
     re.compile(r"\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){7,}\s*\]")),
    ("integer matrix typed into the code", re.compile(r"np\.array\(\s*\[\s*\[\s*\d+\s*,")),
    ("quoted plot ID used as a filter", re.compile(r"(isin\(\s*\[\s*['\"]\d{2,}['\"]|==\s*['\"]\d{2,}['\"])")),
]

# ---------------------------------------------------------------------------
# PUBLISHED NAME
# A citation file exists to publish the author's name, so the name in it is not
# a leak. The username rule cannot tell the two apart on its own, because a
# computer username is often a piece of the person's real name ("ada" inside
# "Ada Lovelace"). These two files may therefore carry the author's name, and
# ONLY where the match sits inside the name as CITATION.cff publishes it.
# Everything else still blocks: the same username in a path, in a source file,
# or anywhere in these files outside the published name.
# The name itself is never written here; it is read from CITATION.cff.
AUTHORSHIP_FILES = {"CITATION.cff", "README.md"}
CITATION_NAME_KEYS = ("given-names", "family-names", "name")

# ---------------------------------------------------------------------------
# COMMON_WORD_TERMS
# Ordinary English and programming words. A private term that is one of these
# is NOT dropped -- it is matched case-sensitively instead of case-insensitively,
# so a variety named ALPHA is still caught when ALPHA is typed into the code,
# while `alpha=3.0` (an ordinary parameter name) no longer raises a false BLOCK.
#
# This is a GENERIC word list. A word appearing here says nothing about whether
# it is a variety in any particular trial, so the list leaks nothing.
#
# Only applied to terms shorter than COMMON_WORD_MAX_LEN, where a collision with
# normal code is likely and the term is too short to be distinctive. Every entry
# is therefore 5 characters or fewer; longer words here would never be used.
COMMON_WORD_MAX_LEN = 6
COMMON_WORD_TERMS = {
    # Greek letters, routinely used as parameter names
    "alpha", "beta", "gamma", "delta", "theta", "kappa", "sigma", "omega",
    # everyday identifiers in scientific Python
    "axis", "data", "eos", "file", "fit", "grid", "index", "key", "label",
    "list", "mask", "max", "mean", "min", "mode", "model", "norm", "path",
    "plot", "rate", "raw", "row", "score", "seed", "size", "sort", "step",
    "sum", "test", "text", "time", "train", "type", "value",
}

THIS_FILE = Path(__file__).resolve()


@dataclass
class Finding:
    level: str      # BLOCK or REVIEW
    rule: str
    where: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Which files would actually be published?
# ---------------------------------------------------------------------------
def git_publishable_files(root: Path) -> list[Path] | None:
    """Tracked files + new files that .gitignore does NOT exclude.
    Returns None if root is not inside a git repo yet."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True, check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [root / p for p in out.split("\0") if p]


def walk_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            files.append(p)
    return files


def git_history_data_files(root: Path) -> set[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "--all", "--pretty=format:", "--name-only"],
            capture_output=True, check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {line for line in out.splitlines()
            if line and Path(line).suffix.lower() in DATA_EXT | FIGURE_EXT | {".ipynb"}}


# ---------------------------------------------------------------------------
# Private terms (e.g. variety names) — the list itself lives OUTSIDE the repo
# ---------------------------------------------------------------------------
def load_terms(terms_file: str | None, tables: list[str], column: str | None) -> set[str]:
    terms: set[str] = set()
    if terms_file:
        for line in Path(terms_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                terms.add(line.strip())
    if tables:
        if not column:
            sys.exit("--terms-from-table needs --column (the column holding the names).")
        import pandas as pd  # only needed for this option
        for t in tables:
            path = Path(t)
            frames = (pd.read_excel(path, sheet_name=None, header=None).values()
                      if path.suffix.lower() in {".xlsx", ".xls"} else [pd.read_csv(path, header=None)])
            for df in frames:
                # the header is not always the first row (e.g. a title row above it),
                # so look for the column name anywhere in the first 10 rows
                for r in range(min(10, len(df))):
                    hits = [c for c in df.columns if str(df.iat[r, c]).strip() == column]
                    if hits:
                        for c in hits:
                            terms.update(str(v).strip() for v in df.iloc[r + 1:, c].dropna())
                        break
        if not terms:
            sys.exit(f"No values found under a '{column}' header in {tables}. "
                     "Check the column name; refusing to run a check that would silently test nothing.")
    # drop things that would match everywhere: pure numbers and very short strings
    return {t for t in terms if len(t) >= 3 and not re.fullmatch(r"[\d.\s]+", t)}


# ---------------------------------------------------------------------------
# The author's own name, as they publish it
# ---------------------------------------------------------------------------
def published_author_names(root: Path) -> list[str]:
    """Names the author deliberately publishes, read from CITATION.cff.

    Returns each name part and every given+family combination, longest first,
    so the longest published form wins when deciding whether a username match
    sits inside a name. Returns [] when there is no readable CITATION.cff, in
    which case nothing is exempt and the username rule behaves as it always did.
    """
    cff = root / "CITATION.cff"
    if not cff.exists():
        return []
    try:
        lines = cff.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    names: set[str] = set()
    given: list[str] = []
    family: list[str] = []
    for line in lines:
        item = line.strip().lstrip("-").strip()          # tolerate "- given-names: ..."
        for key in CITATION_NAME_KEYS:
            if item.lower().startswith(key + ":"):
                value = item.split(":", 1)[1].strip().strip('"').strip("'")
                if not value:
                    continue
                names.add(value)
                if key == "given-names":
                    given.append(value)
                elif key == "family-names":
                    family.append(value)
    for g in given:
        for f in family:
            names.add(f"{g} {f}")
    return sorted(names, key=len, reverse=True)


def inside_published_name(line: str, span: tuple[int, int], names: list[str]) -> bool:
    """True if the matched span lies wholly inside an occurrence of a published
    name on this line. A username that merely shares the line with the author's
    name is NOT exempt; it has to be part of the name itself."""
    start, end = span
    low = line.lower()
    for name in names:
        needle = name.lower()
        pos = low.find(needle)
        while pos != -1:
            if pos <= start and end <= pos + len(needle):
                return True
            pos = low.find(needle, pos + 1)
    return False


def term_pattern(term: str) -> re.Pattern:
    """Whole-word matcher for one private term.

    Ordinary words (COMMON_WORD_TERMS) are matched case-sensitively so that the
    variety ALPHA still blocks while `alpha=3.0` does not. Everything else stays
    case-insensitive, so a variety name typed in any casing is still caught.
    """
    ordinary = len(term) < COMMON_WORD_MAX_LEN and term.casefold() in COMMON_WORD_TERMS
    flags = 0 if ordinary else re.I
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", flags)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def scan(files: list[Path], root: Path, terms: set[str], history: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    user = getpass.getuser()
    user_pat = re.compile(re.escape(user), re.I) if len(user) >= 3 and user not in {"root", "user"} else None
    term_pats = [(t, term_pattern(t)) for t in sorted(terms)]
    author_names = published_author_names(root)

    for f in sorted(set(files)):
        if not f.exists() or f.resolve() == THIS_FILE:
            continue
        rel = f.relative_to(root) if f.is_relative_to(root) else f
        ext = f.suffix.lower()

        if ext in DATA_EXT:
            findings.append(Finding("BLOCK", "data file would be uploaded", str(rel)))
            continue
        if ext in FIGURE_EXT:
            findings.append(Finding("BLOCK", "figure / export would be uploaded (drawn from real data)", str(rel)))
            continue
        if f.stat().st_size > MAX_BYTES:
            findings.append(Finding("BLOCK", "file larger than 1 MB", str(rel), f"{f.stat().st_size/1e6:.1f} MB"))

        if ext == ".ipynb":
            try:
                nb = json.loads(f.read_text(encoding="utf-8"))
                n_out = sum(1 for c in nb.get("cells", []) if c.get("outputs"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                n_out = -1
            if n_out != 0:
                findings.append(Finding("BLOCK", "notebook contains saved outputs", str(rel),
                                        f"{n_out} cells with output" if n_out > 0 else "unreadable notebook"))
            lines = [""]  # outputs already handled; sources scanned below via json
            try:
                lines = ["".join(c.get("source", "")) for c in nb.get("cells", [])]
                lines = "\n".join(lines).splitlines()
            except Exception:
                pass
        elif ext in TEXT_EXT or ext == "":
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                findings.append(Finding("REVIEW", "binary file", str(rel)))
                continue
        else:
            findings.append(Finding("REVIEW", "unrecognised file type", str(rel)))
            continue

        for i, line in enumerate(lines, start=1):
            loc = f"{rel}:{i}"
            snippet = line.strip()[:90]
            for pat in PATH_PATTERNS:
                if pat.search(line):
                    findings.append(Finding("BLOCK", "personal absolute path", loc, snippet))
                    break
            else:
                if user_pat:
                    # The author's published name is not a leaked username.
                    may_be_name = str(rel) in AUTHORSHIP_FILES and bool(author_names)
                    for m in user_pat.finditer(line):
                        if may_be_name and inside_published_name(line, m.span(), author_names):
                            continue
                        findings.append(Finding("BLOCK", "your computer username appears", loc, snippet))
                        break
            for term, pat in term_pats:
                if pat.search(line):
                    findings.append(Finding("BLOCK", "private term (from your terms list)", loc, f"'{term}' in: {snippet}"))
            for label, pat in LITERAL_PATTERNS:
                if pat.search(line):
                    findings.append(Finding("REVIEW", label, loc, snippet))

    for h in sorted(history):
        findings.append(Finding("BLOCK", "data/figure/notebook file exists in git HISTORY", h,
                                 "deleting it is not enough; history must be rewritten or the repo recreated"))
    return findings


def report(findings: list[Finding], mode_note: str, out_path: str | None) -> int:
    blocks = [x for x in findings if x.level == "BLOCK"]
    reviews = [x for x in findings if x.level == "REVIEW"]
    lines = ["# Leak check report", "", mode_note, "",
             f"BLOCK: {len(blocks)}    REVIEW: {len(reviews)}", ""]
    for level, group in (("BLOCK", blocks), ("REVIEW", reviews)):
        if not group:
            continue
        lines.append(f"## {level}")
        by_rule: dict[str, list[Finding]] = {}
        for x in group:
            by_rule.setdefault(x.rule, []).append(x)
        for rule, items in by_rule.items():
            lines.append(f"\n### {rule}  ({len(items)})")
            for x in items:
                lines.append(f"- `{x.where}`" + (f"  {x.detail}" if x.detail else ""))
        lines.append("")
    verdict = ("RESULT: NOT SAFE TO PUSH. Fix every BLOCK item first."
               if blocks else
               "RESULT: no BLOCK items. Read the REVIEW items yourself, then push.")
    lines.append(verdict)
    text = "\n".join(lines)
    print(text)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"\n(report also written to {out_path} — do NOT commit it)")
    return 1 if blocks else 0


# ---------------------------------------------------------------------------
# Sanity check: plant known leaks in a temp folder and confirm every one is caught
# ---------------------------------------------------------------------------
def self_test() -> int:
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "plots.csv").write_text("Disease_Plot_ID,Disease_class\n1,10\n")
        (tmp / "fig.png").write_bytes(b"\x89PNG fake")
        (tmp / "nb.ipynb").write_text(json.dumps({"cells": [
            {"cell_type": "code", "source": ["df.head()"], "outputs": [{"text": "row data"}]}]}))
        (tmp / "leaky.py").write_text(
            'os.chdir("/Users/someone/Desktop/Research Thesis/data")\n'
            "trial = {'Zorbatato': {'scores': [4.0, 10.0]}}\n"
            "cm = np.array([[51, 17, 18], [12, 19, 16]])\n"
            "row = df[df['Disease_Plot_ID'] == '259']\n"
            "vals = [0.61, 0.62, 0.60, 0.59, 0.63, 0.61, 0.62, 0.60]\n"
        )
        (tmp / "clean.py").write_text(
            "from config import DATA_DIR, RANDOM_SEED\n"
            "df = pd.read_csv(DATA_DIR / 'processed' / 'A2_clean_tex_norm.csv')\n"
            "cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)\n"
        )

        # --- the author's published name vs a genuinely leaked username ------
        # Built from the live username so this file never contains a real one.
        user = getpass.getuser()
        pretty = user.capitalize()
        (tmp / "CITATION.cff").write_text(
            "cff-version: 1.2.0\n"
            "authors:\n"
            f'  - given-names: "{pretty}"\n'
            '    family-names: "Testsurname"\n'
        )
        (tmp / "README.md").write_text(
            "## Contact\n"
            f"{pretty} Testsurname\n"                  # published name: allowed
            f"Clone it to /Users/{user}/Desktop/repo\n"  # real path leak: blocked
        )
        (tmp / "notes.py").write_text(f'CACHE = "/tmp/{user}-cache"\n')  # blocked

        # --- a private term that is also an ordinary word --------------------
        # DELTA stands in for any such term. It is matched case-sensitively:
        # the exact-case name blocks, the lowercase parameter name does not.
        (tmp / "params.py").write_text(
            "model = Ridge(delta=3.0)\n"    # ordinary parameter name: allowed
            'GROUP = "DELTA"\n'             # the term itself, exact case: blocked
        )
        # and a term that is NOT an ordinary word stays case-insensitive
        (tmp / "mixedcase.py").write_text("note = 'zorbatato had the worst plots'\n")

        found = scan(walk_files(tmp), tmp, {"Zorbatato", "DELTA"}, set())

        def hit(level, rule_part, where_part):
            return any(x.level == level and rule_part in x.rule and where_part in x.where for x in found)

        checks = [
            ("csv data file blocked", hit("BLOCK", "data file", "plots.csv")),
            ("figure blocked", hit("BLOCK", "figure", "fig.png")),
            ("notebook outputs blocked", hit("BLOCK", "notebook contains saved outputs", "nb.ipynb")),
            ("personal path blocked", hit("BLOCK", "personal absolute path", "leaky.py:1")),
            ("private term blocked", hit("BLOCK", "private term", "leaky.py:2")),
            ("score list flagged", hit("REVIEW", "score list", "leaky.py:2")),
            ("integer matrix flagged", hit("REVIEW", "integer matrix", "leaky.py:3")),
            ("quoted plot ID flagged", hit("REVIEW", "plot ID", "leaky.py:4")),
            ("long number list flagged", hit("REVIEW", "long list", "leaky.py:5")),
            ("clean file NOT flagged", not any("clean.py" in x.where for x in found)),

            # (1) published author name vs leaked username
            ("author name in CITATION.cff not called a username",
             not hit("BLOCK", "username", "CITATION.cff")),
            ("author name in README authorship line not called a username",
             not hit("BLOCK", "username", "README.md:2")),
            ("personal path in README STILL blocked",
             hit("BLOCK", "personal absolute path", "README.md:3")),
            ("username outside an author name STILL blocked",
             hit("BLOCK", "your computer username appears", "notes.py:1")),

            # (2) ordinary-word private terms
            ("ordinary word not flagged as a private term",
             not any("params.py:1" in x.where and "private term" in x.rule for x in found)),
            ("private term in its exact case STILL blocked",
             hit("BLOCK", "private term", "params.py:2")),
            ("non-ordinary term STILL matched case-insensitively",
             hit("BLOCK", "private term", "mixedcase.py:1")),
        ]
        print("SELF-TEST")
        for name, ok in checks:
            print(f"  {'PASS' if ok else 'MISMATCH'}  {name}")
        failed = [n for n, ok in checks if not ok]
        print("\nAll checks passed: the scanner can be trusted." if not failed
              else f"\n{len(failed)} check(s) failed: do NOT trust this scanner until fixed.")
        return 0 if not failed else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Check a repo for data leaks before pushing.")
    ap.add_argument("paths", nargs="*", help="specific files to scan (default: whole repo)")
    ap.add_argument("--root", default=str(THIS_FILE.parent.parent), help="repo root (default: parent of tools/)")
    ap.add_argument("--private-terms", help="text file of private terms, one per line, kept OUTSIDE the repo")
    ap.add_argument("--terms-from-table", action="append", default=[],
                    help="csv/xlsx kept OUTSIDE the repo; names in --column become private terms")
    ap.add_argument("--column", help="column holding private names, e.g. Variety or Variety_ID")
    ap.add_argument("--report", help="also write the report to this file (keep it out of the repo)")
    ap.add_argument("--self-test", action="store_true", help="run the built-in sanity check and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    root = Path(args.root).resolve()
    terms = load_terms(args.private_terms, args.terms_from_table, args.column)

    if args.paths:
        files = [Path(p).resolve() for p in args.paths]
        history: set[str] = set()
        note = f"Mode: explicit files ({len(files)})."
    else:
        files = git_publishable_files(root)
        if files is None:
            files = walk_files(root)
            history = set()
            note = ("Mode: NOT a git repo yet, so .gitignore is not applied and every file in the "
                    "folder is scanned. Expect data/ files to show up here; run again after `git init`.")
        else:
            history = git_history_data_files(root)
            note = f"Mode: git — scanning the {len(files)} files git would publish, plus commit history."
    note += f" Private terms loaded: {len(terms)}."
    return report(scan(files, root, terms, history), note, args.report)


if __name__ == "__main__":
    sys.exit(main())
