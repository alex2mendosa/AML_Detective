"""
QUALITY CHECK TEST - judge step.

Reads test_companies_quality.json after a run and asks a separate LLM, per company:
did the agent identify the risk (or correctly find nothing)?  Judged on the risk
signal only - not on specific facts, dates, amounts or URLs.

Nothing here touches the production package.

Usage (from the repo root):
    python test_cases/test_quality/quality_check_judge.py
    python test_cases/test_quality/quality_check_judge.py --model gpt-4.1
"""
import argparse, json, os, shutil, sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH = REPO / "test_cases" / "test_quality" / "test_companies_quality.json"

from dotenv import load_dotenv
load_dotenv(REPO / ".env", override=True)
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class Verdict(BaseModel):
    identified: bool = Field(description="True if the agent's conclusion matches the expected risk signal.")
    comment: str = Field(description="One short sentence explaining the verdict.")


JUDGE_PROMPT = """You grade an automated AML adverse-media agent. You are given, for one company:
- the expected label: "risk" (the company is genuinely involved in criminal activity, enforcement action or serious reputational problems) or "clean" (nothing significant known),
- a short note describing the known ground truth,
- the conclusion the agent produced (in Romanian).

Answer ONE question: did the agent arrive at the right risk signal?

Rules:
- Judge the SIGNAL ONLY. Ignore wrong dates, amounts, names of regulators, missing details or extra findings.
- expected "risk": identified = true if the conclusion flags meaningful risk, adverse findings, enhanced due diligence, suspension or termination. It is NOT required to name the specific case from the note.
- expected "clean": identified = true if the conclusion reports no significant adverse information and does not recommend terminating or suspending the relationship.
- If the expected label is "risk" and the agent accurately reports that the case ended in acquittal, dismissal or closure, that still counts as identified = true - it found and assessed the matter.
- If the agent found adverse information about a DIFFERENT company with a similar name, identified = false and say so in the comment (misattribution).
- An empty or missing conclusion is identified = false.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    args = ap.parse_args()

    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    todo = [c for c in gt["companies"] if c.get("our_conclusion")
            and c.get("run_status", "valid") == "valid"]   # never grade a run whose search was broken
    if not todo:
        sys.exit("No companies have our_conclusion filled in - run quality_check_runner.py first.")

    llm = ChatOpenAI(model=args.model, api_key=os.environ["OPENAI_API_KEY"],
                     temperature=0, max_retries=3, timeout=120).with_structured_output(Verdict)

    shutil.copy(GROUND_TRUTH, GROUND_TRUTH.with_suffix(".json.bak"))
    for c in todo:
        payload = (f"Company: {c['name']}\n"
                   f"Expected label: {c['expected']}\n"
                   f"Known ground truth: {c['expected_note']}\n\n"
                   f"Agent conclusion (Romanian):\n{c['our_conclusion']}\n\n"
                   f"Agent summary (Romanian):\n{c.get('our_rezumat', '')}\n\n"
                   f"Agent recommendation: {c.get('our_recomandare', '')}\n"
                   f"Agent risk score (0-100): {c.get('our_scor_risc')}")
        v = llm.invoke([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=payload)])
        c["judge_match"] = v.identified
        c["judge_comment"] = v.comment
        print(f"  {c['name']:26} expected={c['expected']:6} score={str(c.get('our_scor_risc')):4} "
              f"-> {'OK  ' if v.identified else 'MISS'} | {v.comment[:90]}")

    risk = [c for c in todo if c["expected"] == "risk"]
    clean = [c for c in todo if c["expected"] == "clean"]
    hits = sum(1 for c in risk if c["judge_match"])
    fps = sum(1 for c in clean if not c["judge_match"])
    gt["last_judge"] = {
        "test_type": "quality_check", "date": date.today().isoformat(), "model": args.model,
        "judged": len(todo),
        "risk_rows": len(risk), "risk_identified": hits,
        "clean_rows": len(clean), "false_positives": fps,
    }
    GROUND_TRUTH.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[quality_check] judged {len(todo)} | risk identified {hits}/{len(risk)} | "
          f"false positives {fps}/{len(clean)}")
    print(f"[quality_check] written to {GROUND_TRUTH.name} (backup: .json.bak)")


if __name__ == "__main__":
    main()
