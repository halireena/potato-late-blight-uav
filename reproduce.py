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

   A failing script does NOT stop the ones after it. Only real dependencies are
   honoured: 01 writes data/processed/, which every later script reads, so if 01
   fails the rest cannot run and are marked SKIPPED. Scripts 02 onwards each read
   those tables and none reads another's output, so one of them failing says
   nothing about the others and must not silence them. Every script's result is
   reported on its own.
4. Verdict. Collects every check into one answer, REPRODUCED or NOT REPRODUCED,
   and saves a report to outputs/reproducibility_report.txt, stamped with the
   date, the exact code version (git commit) and the environment. NOT REPRODUCED
   still lists which scripts passed, because "one number is wrong" and "nothing
   works" are different situations and the report should not blur them.

The report also carries a separate, clearly fenced DIAGNOSTICS section for output
that has NO authorised value behind it and is therefore not checked against
anything. Those numbers are pointers for investigation, not results, and the
section exists so they cannot be mistaken for checked figures.

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

# 01 writes data/processed/, which every later script reads. That is the only
# real dependency in the pipeline: 02 to 07 each read those tables and none
# reads another's output. So a failure in 02 must not stop 03 from running.
BUILDS_PROCESSED = "01_build_plot_table.py"

# Scripts whose output is diagnostic: they check nothing, because no authorised
# value exists for anything they print. Reported separately so their numbers are
# never read as verified.
DIAGNOSTIC_SCRIPTS = {"07_selection_leakage_check.py"}

# Blocks inside otherwise-checked scripts that are deliberately NOT checked,
# for the same reason. Listed by hand so the report names them exactly.
UNVERIFIED_BLOCKS = [
    ("09_ceiling_and_search.py", "the moderate learning curve at the full class size",
     "at that size nothing is sampled and only row order varies, which changes fold "
     "membership and so the predictions; the point is not determined to three decimals"),
    ("09_ceiling_and_search.py", "whether the two learning curves are flat",
     "it is an observation about the curves, not a recorded value, and it bears on a "
     "Discussion claim"),
    ("09_ceiling_and_search.py", "the two moderate-class learning curves on record",
     "0.347-to-0.369 and 0.395-to-0.440 come from different models, so neither is the "
     "value for the other"),
    ("03_architectures.py / 08_remaining_permutations.py",
     "one-vs-rest ensemble, within-flight macro-F1",
     "the notebook leaves its Platt calibration unseeded, so the recorded procedure does not "
     "determine the value; ten unseeded runs spanned 0.300 to 0.323"),
    ("08_remaining_permutations.py", "MLP cascade, test-flight p-value",
     "that row uses 200 shuffles, where the standard error on a p near 0.69 is about 0.033, "
     "wider than the comparison tolerance"),
    ("08_remaining_permutations.py", "single-stage three-class within-flight, both variants",
     "Table S2 records 0.380 and 04 verifies 0.358, on different feature sets with "
     "different aggregations"),
    ("07_selection_leakage_check.py", "every number it prints",
     "the leak-free texture selection, and the cascade refitted with leak-free "
     "features and with no texture at all"),
    ("05_supporting_statistics.py", "prevalence-shift resampling, schemes A and B",
     "two recorded values exist (0.293 and 0.370) and the scheme is not what "
     "separates them"),
    ("04_permutation_tests.py", "ordinal regression, within-flight permutation test",
     "two real scores are on record for it (0.295 and 0.308)"),
    ("03_architectures.py", "ordinal regression, within-flight macro-F1",
     "two values are on record for it, 0.308 and 0.295"),
]


def prerequisites(script_name):
    """Scripts that must have PASSED before this one can run."""
    return [] if script_name == BUILDS_PROCESSED else [BUILDS_PROCESSED]


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
    results = {}                      # script name -> how it went
    verified_sections, diagnostic_sections = [], []
    if not missing:
        scripts = sorted((ROOT / "src").glob("[0-9][0-9]_*.py"))
        for i, script in enumerate(scripts, start=1):
            name = script.name
            blocked = [d for d in prerequisites(name)
                       if results.get(d, {}).get("status") != "PASSED"]
            if blocked:
                # Only skip when something this script genuinely needs did not pass.
                results[name] = {"status": "SKIPPED", "code": None, "ok": 0, "bad": [],
                                 "why": "needs " + ", ".join(blocked)}
                print(f"Step 3/3  Skipping {name} ({i} of {len(scripts)}): "
                      + results[name]["why"])
                continue

            print(f"Step 3/3  Running {name} ({i} of {len(scripts)})...")
            code, output = run_script(script)
            ok, bad = parse_checks(output)
            diagnostic = name in DIAGNOSTIC_SCRIPTS
            if not diagnostic:
                all_ok += ok
                all_bad += bad
            status = "PASSED" if code == 0 and not bad else "FAILED"
            results[name] = {"status": status, "code": code, "ok": len(ok), "bad": bad,
                             "why": "", "diagnostic": diagnostic}

            section = [LINE, f"{name}  (exit code {code}, {status})", LINE, output.rstrip(), ""]
            (diagnostic_sections if diagnostic else verified_sections).extend(section)

            if "DIFFERS" in output:
                failures.append(f"{name}: rebuilt table differs from the existing processed table")
            if code != 0:
                failures.append(f"{name}: stopped with exit code {code}")

    # Per-script results, so one failure cannot be read as everything failing
    if results:
        report += [LINE, "SCRIPT RESULTS", LINE]
        width = max(len(n) for n in results)
        for name, r in results.items():
            tag = " (diagnostic, checks nothing)" if r.get("diagnostic") else ""
            detail = (r["why"] if r["status"] == "SKIPPED"
                      else f"{r['ok']} check(s) matched"
                           + (f", {len(r['bad'])} did not" if r["bad"] else ""))
            report.append(f"  [{r['status']:7s}] {name:{width}s}  {detail}{tag}")
        report.append("")
    report += verified_sections

    if all_bad:
        failures.append(f"{len(all_bad)} sanity check(s) did not match")
    skipped_scripts = [n for n, r in results.items() if r["status"] == "SKIPPED"]
    if skipped_scripts:
        failures.append("script(s) not run because a prerequisite failed: "
                        + ", ".join(skipped_scripts))

    # Diagnostics, fenced off from everything that is actually checked
    fence = "#" * 70
    report += [fence,
               "DIAGNOSTICS - NOT CHECKED AGAINST ANYTHING",
               fence,
               "Everything in this section is UNVERIFIED. No authorised value exists for any",
               "number below, so nothing here has been compared with the thesis and nothing",
               "here contributes to the verdict. These are pointers for investigation, not",
               "results, and they must not be quoted as if they had been checked.",
               ""]
    report.append("Blocks that are deliberately printed without a check:")
    for script, what, why in UNVERIFIED_BLOCKS:
        report.append(f"  - {script}: {what}")
        report.append(f"      because {why}")
    report.append("")
    if diagnostic_sections:
        report += ["Full output of the diagnostic-only script(s):", ""] + diagnostic_sections
    else:
        report += ["No diagnostic-only script ran.", ""]

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
    if not reproduced and results:
        passed = [n for n, r in results.items() if r["status"] == "PASSED"]
        verdict.append("")
        verdict.append(f"  {len(all_ok)} check(s) DID match. Scripts that passed in full "
                       f"({len(passed)} of {len(results)}):")
        for n in passed:
            tag = "  (diagnostic, checks nothing)" if results[n].get("diagnostic") else ""
            verdict.append(f"    [PASSED] {n}{tag}")
        for n, r in results.items():
            if r["status"] == "FAILED":
                verdict.append(f"    [FAILED] {n}: {len(r['bad'])} check(s) did not match"
                               if r["bad"] else f"    [FAILED] {n}: exit code {r['code']}")
            elif r["status"] == "SKIPPED":
                verdict.append(f"    [SKIPPED] {n}: {r['why']}")
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
