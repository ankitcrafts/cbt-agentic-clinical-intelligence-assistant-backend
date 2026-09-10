import sys
from pathlib import Path

# Add backend root to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.core.database import init_db

print("--- Testing Backend Database Initialization & Seeding ---")
init_db()

client = TestClient(app)

print("\n--- 1. Testing GET /api/scheduling/patients ---")
res = client.get("/api/scheduling/patients")
print(f"Status: {res.status_code}")
data = res.json()
print(f"Success: {data.get('success')}, Count: {data.get('count')}")
assert res.status_code == 200
assert data.get("count", 0) > 0

print("\n--- 2. Testing GET /api/scheduling/doctors ---")
res_docs = client.get("/api/scheduling/doctors")
print(f"Status: {res_docs.status_code}")
docs_data = res_docs.json()
print(f"Success: {docs_data.get('success')}, Doctors Count: {docs_data.get('count')}")
assert res_docs.status_code == 200

print("\n--- 3. Testing POST /api/scheduling/patients (Register New Patient) ---")
new_pat_payload = {
    "name": "David Miller",
    "age": 46,
    "gender": "M",
    "phone": "+1 (555) 998-1122",
    "reason": "Exertional dyspnea and hypertension check",
    "condition": "Cardiovascular screening",
    "visit_type": "New Patient",
    "vitals": {"bp": "135/85 mmHg", "hr": "78 bpm", "pain": "2/10", "spo2": "98%", "temp": "36.7 °C"},
    "preferred_doctor_name": "Dr. Yuki Tanaka",
    "is_preferred_doctor_choice": True
}
res_create = client.post("/api/scheduling/patients", json=new_pat_payload)
print(f"Status: {res_create.status_code}")
pat_created = res_create.json()
print(f"Created Patient ID: {pat_created.get('data', {}).get('id')}, Name: {pat_created.get('data', {}).get('name')}")
assert res_create.status_code == 200
created_pat_id = pat_created.get("data", {}).get("id")

print("\n--- 4. Testing POST /api/scheduling/assign (Doctor & Time Slot Allocation) ---")
assign_payload = {
    "patient_id": created_pat_id,
    "doctor_id": "DOC-001",
    "doctor_name": "Dr. Yuki Tanaka",
    "time_slot": "10:30 AM",
    "duration": "30 min",
    "is_preferred_choice": True,
    "preferred_doctor_name": "Dr. Yuki Tanaka",
    "visit_type": "Consultation",
    "booking_notes": "Assigned to Dr. Yuki Tanaka at 10:30 AM per patient preference."
}
res_assign = client.post("/api/scheduling/assign", json=assign_payload)
print(f"Status: {res_assign.status_code}")
assign_data = res_assign.json()
print(f"Assign Message: {assign_data.get('message')}")
assert res_assign.status_code == 200

print("\n--- 5. Testing GET /api/scheduling/queue/DOC-001 (Doctor Dynamic Queue) ---")
res_queue = client.get("/api/scheduling/queue/DOC-001")
print(f"Status: {res_queue.status_code}")
queue_data = res_queue.json()
print(f"Doctor Queue Count: {queue_data.get('queueCount')}")
assert res_queue.status_code == 200
assert any(p.get("id") == created_pat_id for p in queue_data.get("data", []))

print("\n--- 6. Testing GET /api/scheduling/patients/#00847261/sessions (Patient Past Records) ---")
res_sess = client.get("/api/scheduling/patients/%2300847261/sessions")
print(f"Status: {res_sess.status_code}")
sess_data = res_sess.json()
print(f"Patient #00847261 Past Sessions Count: {sess_data.get('count')}")
assert res_sess.status_code == 200

print("\n=== ALL END-TO-END SCHEDULING FLOW TESTS PASSED SUCCESSFULLY! ===")
