"""
reproduce.py — the ONE command that answers "does this code reproduce the thesis?"

    python reproduce.py

What it does, in order
----------------------
1. Environment. Compares your Python and package versions with requirements.txt.
   Different versions can shift results slightly, so a mismatch is reported.
2. Inputs. Checks every raw input file is present before anything runs.
3. Pipeline. Runs every script in src/ in numbered order. Each script ends with
   its own sanity check against numbers printed in the original notebook.
4. Verdict. Collects every check into one answer, REPRODUCED or NOT REPRODUCED,
   and saves a report to outputs/reproducibility_report.txt, stamped with the
   date, the exact code version (git commit) and the environment.

Exit code 0 = reproduced, 1 = not reproduced.
"""
import datetime
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, OUT_DIR  # noqa: E402

EXPECTED_PYTHON = (3, 9)
RAW_INPUTS = [
    "ALL_stats_1_JB_jack_data.csv",
    "ALL_stats_2_JB_jack_data.csv",
    "final_disease_data.csv",
    "variety_and_plot_IDs.xlsx",
    "variety_spelling_fixes.csv",
    "texture_features_selected.csv",
    "texture_features_percentiles.csv",
    "sharma_genotype_reduced_recoded_NA.csv",
]
# Inputs that unlock extra statistics but are not required. A missing one is
# reported, and the script that needs it skips only the statistics that use it;
# it never fails the run and is never substituted with another file.
OPTIONAL_INPUTS = [
    "sharma_SNP_positions.xlsx",      # SNP marker positions; LD pruning and both fusion tests
]
LINE = "=" * 70


def check_environment():
    """Return (list of report lines, list of mismatches)."""
    lines, problems = [], []
    py = sys.version_info[:2]
    lines.append(f"Python {platform.python_version()} (expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.x)")
    if py != EXPECTED_PYTHON:
        problems.append(f"Python {py[0]}.{py[1]} instead of {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}")
    for raw in (ROOT / "requirements.txt").read_text().splitlines():
        raw = raw.split("#")[0].strip()
        if "==" not in raw:
            continue
        name, wanted = raw.split("==")
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "NOT INSTALLED"
        ok = have == wanted
        lines.append(f"  [{'OK' if ok else 'DIFFERENT'}] {name:18s} installed {have:14s} expected {wanted}")
        if not ok:
            problems.append(f"{name} {have} instead of {wanted}")
    return lines, problems


def git_state():
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + (" (with UNCOMMITTED changes: this run is not tied to a commit)" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not a git repository"


def parse_checks(output):
    """Pull every [OK] / [MISMATCH] line a script printed."""
    ok = [l.strip() for l in output.splitlines() if re.search(r"\[OK\]", l)]
    bad = [l.strip() for l in output.splitlines() if re.search(r"\[MISMATCH\]", l)]
    return ok, bad


def run_script(path):
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, cwd=ROOT)
    return result.returncode, result.stdout + result.stderr


def main():
    report = [LINE, "REPRODUCIBILITY CHECK", LINE,
              f"Date:        {datetime.datetime.now():%Y-%m-%d %H:%M}",
              f"Code:        git commit {git_state()}",
              f"Platform:    {platform.platform()}",
              f"Data folder: {DATA_DIR}", ""]
    failures = []

    # 1. Environment
    print("Step 1/3  Checking environment...")
    env_lines, env_problems = check_environment()
    report += ["ENVIRONMENT"] + env_lines + [""]

    # 2. Inputs
    print("Step 2/3  Checking input files...")
    missing = [f for f in RAW_INPUTS if not (DATA_DIR / f).exists()]
    report += ["INPUT FILES"] + [f"  [{'MISSING' if f in missing else 'OK'}] {f}" for f in RAW_INPUTS]
    absent_optional = [f for f in OPTIONAL_INPUTS if not (DATA_DIR / f).exists()]
    report += ["OPTIONAL INPUTS (absence skips statistics, it does not fail the run)"]
    report += [f"  [{'ABSENT' if f in absent_optional else 'OK'}] {f}" for f in OPTIONAL_INPUTS] + [""]
    if missing:
        failures.append(f"missing input files: {', '.join(missing)} (see data/README.md)")

    # 3. Pipeline
    all_ok, all_bad = [], []
    if not missing:
        scripts = sorted((ROOT / "src").glob("[0-9][0-9]_*.py"))
        for i, script in enumerate(scripts, start=1):
            print(f"Step 3/3  Running {script.name} ({i} of {len(scripts)})...")
            code, output = run_script(script)
            ok, bad = parse_checks(output)
            all_ok += ok
            all_bad += bad
            report += [LINE, f"{script.name}  (exit code {code})", LINE, output.rstrip(), ""]
            if "DIFFERS" in output:
                failures.append(f"{script.name}: rebuilt table differs from the existing processed table")
            if code != 0:
                failures.append(f"{script.name}: stopped with exit code {code}")
                break                      # later scripts depend on this one

    if all_bad:
        failures.append(f"{len(all_bad)} sanity check(s) did not match")

    # Verdict
    reproduced = not failures and len(all_ok) > 0
    verdict = [LINE,
               ("REPRODUCED: all {} sanity checks match the thesis numbers.".format(len(all_ok)) if reproduced
                else "NOT REPRODUCED"),
               LINE]
    for f in failures:
        verdict.append(f"  - {f}")
    for b in all_bad:
        verdict.append(f"    {b}")
    if env_problems:
        verdict.append("")
        verdict.append("Environment differs from requirements.txt"
                       + (" (results still matched):" if reproduced else " (a likely cause):"))
        verdict += [f"  - {p}" for p in env_problems]
    report += verdict

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "reproducibility_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print()
    print("\n".join(verdict))
    print(f"\nFull report: {report_path}")
    sys.exit(0 if reproduced else 1)


if __name__ == "__main__":
    main()
