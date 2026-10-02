import json
import time
from pathlib import Path

from app.services.pipeline_service import ClinicalPipeline

time.sleep(15)

pipe = ClinicalPipeline()
conv = json.loads(Path("output/20260820_132412_conversation.json").read_text())

print("=== ROUND 0 (force_follow_up for test) ===")
r0 = pipe.run_from_conversation(conv, visit_id="visit-followup-demo2", force_follow_up=True)
print("status:", r0.get("status"))
print("confidence:", r0.get("confidence"))
print("questions:")
for q in r0.get("follow_up_questions") or []:
    print(" -", q.get("id"), q.get("question"))

answers = []
for q in r0.get("follow_up_questions") or []:
    qtext = q.get("question", "").lower()
    answers.append(
        {
            "question_id": q["id"],
            "answer": (
                "No fever, no back pain, and I am not pregnant."
                if any(k in qtext for k in ("pregnan", "fever", "back", "flank"))
                else "Symptoms started yesterday and are mild only. No antibiotics recently."
            ),
        }
    )
if not answers:
    answers = [{"question_id": "q1", "answer": "No fever or flank pain. Not pregnant."}]

print("\n=== ROUND 1 apply_follow_up ===")
r1 = pipe.apply_follow_up("visit-followup-demo2", answers=answers)
print("status:", r1.get("status"))
print("iteration:", r1.get("follow_up_iteration"))
print(
    "prev_conf:",
    r1.get("previous_confidence"),
    "new_conf:",
    r1.get("confidence"),
    "improved:",
    r1.get("confidence_improved"),
)
print("follow_up_questions:", r1.get("follow_up_questions"))
top = (r1.get("differential_diagnosis") or [{}])[0]
print("top dx:", top.get("name"), top.get("confidence"))
print("soap plan:", (r1.get("soap") or {}).get("plan", "")[:120])
print("pdf:", r1.get("soap_filename"))

print("\n=== ROUND 2 should fail ===")
try:
    pipe.apply_follow_up("visit-followup-demo2", answers=answers)
    print("ERROR: should have failed")
except ValueError as e:
    print("OK rejected:", e)
print("SMOKE_OK")
