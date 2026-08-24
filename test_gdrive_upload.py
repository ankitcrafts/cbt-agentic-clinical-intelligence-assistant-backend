import os
import glob
from app.services.gdrive_service import gdrive_service

print("[Test] Testing Google Drive Upload...")
files = glob.glob("app/uploads/*.wav")
if not files:
    print("[Error] No .wav files found in app/uploads/")
else:
    test_file = files[0]
    print(f"[Test] Uploading file: {test_file}")
    res = gdrive_service.upload_file(test_file)
    print("\n================== RESULT ==================")
    print("Success:", res.get("success"))
    if res.get("success"):
        print("Google Drive File ID:", res.get("file_id"))
        print("Google Drive Link:", res.get("web_view_link"))
    else:
        print("Error Details:", res.get("error"))
    print("============================================")
