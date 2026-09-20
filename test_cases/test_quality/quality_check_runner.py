"""
QUALITY CHECK TEST - Pipeline 2 (deep research), PRODUCTION version.

Question answered: does the agent identify risk for companies known to have it,
and stay quiet for companies that do not?  Not a security test - the prompt-injection
tests are separate and use synthetic pages.

PRODUCTION IS NOT MODIFIED AND NOT POLLUTED:
  - the real PROD module is imported as a package and used as-is,
  - Config.AD_MEDIA_DIR and Config.RESEARCH_LOG are redirected into test_cases/runs/...,
  - run_osint_agent() is called directly, so the __main__ block and the Oracle upload
    never execute.

Usage (from the repo root):
    python test_cases/test_quality/quality_check_runner.py --smoke           # 2 companies
    python test_cases/test_quality/quality_check_runner.py                   # all companies
    python test_cases/test_quality/quality_check_runner.py --only "AIRROCK SOLUTIONS" "VITASANMAX"
"""
import argparse, json, os, shutil, sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TEST_DIR = REPO / "test_cases" / "test_quality"
GROUND_TRUTH = TEST_DIR / "test_companies_quality.json"
TEST_TYPE = "quality_check"
VERSION = "prod"

sys.path.insert(0, str(REPO / "to_deploy"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env", override=True)


def build_rows(companies):
    """PROD skips rows with an empty IDENTIFYCODE, so companies without a real IDNO
    get a clearly fake placeholder. ID must be numeric - it names the output file."""
    rows = []
    for i, c in enumerate(companies, start=1):
        rows.append({
            "ID": 900 + i,
            "IDENTIFYCODE": c["idno"] or f"QC-TEST-{i:02d}",
            "SNAME": c["name"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run one risk + one clean company")
    ap.add_argument("--only", nargs="*", help="run only these company names")
    ap.add_argument("--results-per-query", type=int, default=5)
    args = ap.parse_args()

    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    companies = gt["companies"]

    if args.smoke:
        selected = [next(c for c in companies if c["name"] == "AIRROCK SOLUTIONS"),
                    next(c for c in companies if c["name"] == "VITASANMAX")]
    elif args.only:
        wanted = {n.lower() for n in args.only}
        selected = [c for c in companies if c["name"].lower() in wanted]
        if not selected:
            sys.exit(f"No match for {args.only}")
    else:
        selected = companies

    tag = "smoke" if args.smoke else ("subset" if args.only else "full")
    run_dir = TEST_DIR / "runs" / f"{date.today().isoformat()}_{VERSION}_{tag}"
    reports_dir = run_dir / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)          # PROD wipes this dir itself; do it here too for a clean start
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- import PROD as-is, then redirect its outputs OUT of to_deploy/ ---
    import osint_agent.step_2_osint_agent_aml_summary as prod
    prod.Config.AD_MEDIA_DIR = reports_dir          # never point this at test_cases/ - PROD rmtree's it
    prod.Config.RESEARCH_LOG = run_dir / "aml_research.log"

    rows = build_rows(selected)
    print(f"[{TEST_TYPE}] version={VERSION} companies={len(rows)} -> {run_dir}")
    for r in rows:
        print(f"    ID={r['ID']} IDNO={r['IDENTIFYCODE']:<16} {r['SNAME']}")

    prod.run_osint_agent(
        contragents=rows,
        openai_url=os.getenv("OPENAI_URL"),          # unset locally = direct to OpenAI
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_url=os.getenv("TAVILY_URL"),          # unset locally = direct to Tavily
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        serp_api_key=os.getenv("SERP_GOOGLE_API_KEY"),
        NUM_RESULTS_PER_QUERY=args.results_per_query,
    )

    # --- collect results back into the ground-truth file ---
    by_id = {}
    for f in reports_dir.glob("*.json"):
        row = json.loads(f.read_text(encoding="utf-8"))
        by_id[str(row.get("contragentid"))] = row

    # A dead search API produces empty reports that look exactly like genuine clean results,
    # so refuse to mark this run valid if the log shows the provider cut us off.
    log_path = run_dir / "aml_research.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    credit_errors = log_text.count("OUT OF CREDITS")
    run_status = "valid" if credit_errors == 0 else (
        f"INVALID - {credit_errors} out-of-credits errors in the log; search was degraded. Re-run required.")
    if credit_errors:
        print("")
        print(f"!!! {credit_errors} OUT OF CREDITS errors - results are NOT valid !!!")
        print("")

    shutil.copy(GROUND_TRUTH, GROUND_TRUTH.with_suffix(".json.bak"))
    filled = 0
    for c, r in zip(selected, rows):
        row = by_id.get(str(r["ID"]))
        report = (row or {}).get("aml_report") or {}
        if not report:
            c["our_conclusion"] = "NO REPORT PRODUCED"
            c["our_scor_risc"] = None
        else:
            c["our_conclusion"] = report.get("concluzie_finala", "")
            c["our_rezumat"] = report.get("rezumat_analiza", "")
            c["our_recomandare"] = report.get("recomandare_relatie_afaceri", "")
            c["our_scor_risc"] = report.get("scor_risc")
            filled += 1
        c["run_status"] = run_status
        c["run"] = {"test_type": TEST_TYPE, "version": VERSION,
                    "run_dir": str(run_dir.relative_to(REPO)), "date": date.today().isoformat()}
    gt["last_run"] = {"test_type": TEST_TYPE, "version": VERSION, "tag": tag,
                      "date": date.today().isoformat(), "companies": len(rows),
                      "reports_written": filled, "run_dir": str(run_dir.relative_to(REPO))}
    GROUND_TRUTH.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[{TEST_TYPE}] reports: {filled}/{len(rows)} | results written to {GROUND_TRUTH.name} (backup: .json.bak)")
    for c in selected:
        print(f"    {c['name']:26} expected={c['expected']:6} scor_risc={c['our_scor_risc']}")


if __name__ == "__main__":
    main()
