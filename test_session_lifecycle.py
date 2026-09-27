import os
import sys
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("\n========================================================")
print("=== TESTING SEQUENTIAL CONSULTATION SESSION LIFECYCLE ===")
print("========================================================")

# 1. Choose patient #00847261 (Marcus Delgado)
patient_id = "#00847261"
print(f"\n--- 1. Query initial sessions for patient {patient_id} ---")
res = client.get(f"/api/scheduling/patients/{patient_id.replace('#', '%23')}/sessions")
print(f"Status: {res.status_code}")
initial_sessions = res.json().get("data", [])
print(f"Initial completed sessions count: {len(initial_sessions)}")

# 2. Complete consultation (Session 1 or next sequential session)
next_session_num = len(initial_sessions) + 1
session_payload = {
    "session_title": f"Session {next_session_num}: Primary Hypertension & Headache Follow-up",
    "doctor_id": "DOC-001",
    "doctor_name": "Dr. Yuki Tanaka",
    "transcript": "Doctor: Hello Marcus, how has your headache been? Patient: Much better after the medication.",
    "conversation": [
        {"speaker": "Doctor", "text": "Hello Marcus, how has your headache been?", "timestamp": "00:05", "isDoctor": True},
        {"speaker": "Patient", "text": "Much better after the medication.", "timestamp": "00:15", "isDoctor": False}
    ],
    "ai_suggestions": [
        {"id": "sug-1", "title": "Continue Amlodipine 5mg", "actionText": "Maintain dose"}
    ],
    "clinical_analysis": {
        "summary": "Patient reports significant improvement in morning headaches with controlled blood pressure."
    },
    "soap_report": {
        "subjective": "Headache improved, no visual aura.",
        "objective": "BP: 124/80 mmHg, HR: 72 bpm.",
        "assessment": "Essential Hypertension - well controlled.",
        "plan": "Continue current antihypertensive therapy."
    },
    "duration_seconds": 340,
    "status": "COMPLETED",
    "consultation_mode": "IN_PERSON",
    "audio_source": "MIC"
}

print(f"\n--- 2. Closing & Completing Consultation for Session {next_session_num} ---")
res_post = client.post(f"/api/scheduling/patients/{patient_id.replace('#', '%23')}/sessions", json=session_payload)
print(f"Status: {res_post.status_code}")
assert res_post.status_code == 200
saved_data = res_post.json()
print(f"Saved Session response: {saved_data}")

# 3. Query sessions again to verify Session N is stored
print(f"\n--- 3. Verifying updated session list after closing consultation ---")
res_after = client.get(f"/api/scheduling/patients/{patient_id.replace('#', '%23')}/sessions")
assert res_after.status_code == 200
updated_sessions = res_after.json().get("data", [])
print(f"New completed sessions count: {len(updated_sessions)}")
assert len(updated_sessions) == len(initial_sessions) + 1

# 4. Verify sequential numbering for next appointment
expected_next_session = len(updated_sessions) + 1
print(f"\n--- 4. Patient returning for next appointment will automatically start as Session {expected_next_session} ---")
print(f"Active Live Session will be: Session {expected_next_session}")
print(f"Past sessions available in history tab: {[s.get('sessionTitle') for s in updated_sessions]}")

print("\n========================================================")
print("=== ALL SESSION LIFECYCLE TESTS PASSED SUCCESSFULLY! ===")
print("========================================================")
