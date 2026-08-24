import os
import json
import base64
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_ENV = PROJECT_ROOT / ".env"
DEFAULT_FOLDER_ID = "1W0jt4Bb4RJ8ALo53zpzIAdahRfzKCmwi"


def _get_current_config():
    load_dotenv(ROOT_ENV, override=True)
    return {
        "folder_id": os.getenv("GDRIVE_FOLDER_ID", DEFAULT_FOLDER_ID).strip().strip('"').strip("'"),
        "webhook_url": os.getenv("GDRIVE_WEBHOOK_URL", "").strip().strip('"').strip("'"),
        "cred_file": os.getenv("GDRIVE_CREDENTIALS_FILE", "service_account.json").strip().strip('"').strip("'"),
        "sa_json": os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip(),
    }


class GoogleDriveService:
    def __init__(self):
        self.client = None
        self._init_client()

    def _get_credentials_path(self) -> Optional[Path]:
        """Check possible locations for the credentials json file."""
        cfg = _get_current_config()
        candidates = [
            Path(cfg["cred_file"]),
            PROJECT_ROOT / cfg["cred_file"],
            PROJECT_ROOT / "app" / cfg["cred_file"],
            PROJECT_ROOT / "service_account.json",
            PROJECT_ROOT / "token.json",
            PROJECT_ROOT / "client_secret.json",
        ]
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _init_client(self):
        """Initializes Google Drive v3 client with credentials."""
        try:
            from google.oauth2 import service_account
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            cfg = _get_current_config()
            scopes = ["https://www.googleapis.com/auth/drive"]
            creds = None

            # 1. Check for User OAuth token.json
            token_path = PROJECT_ROOT / "token.json"
            if token_path.exists():
                print(f"[GDrive] Loading user OAuth credentials from: {token_path}")
                creds = Credentials.from_authorized_user_file(str(token_path), scopes)

            # 2. Check for Service Account file
            if not creds:
                cred_path = self._get_credentials_path()
                if cred_path and "service_account" in cred_path.name:
                    print(f"[GDrive] Loading service account from: {cred_path}")
                    creds = service_account.Credentials.from_service_account_file(
                        str(cred_path), scopes=scopes
                    )

            # 3. Check for Service Account JSON string in env
            if not creds and cfg["sa_json"]:
                print("[GDrive] Loading credentials from GDRIVE_SERVICE_ACCOUNT_JSON env variable")
                info = json.loads(cfg["sa_json"])
                creds = service_account.Credentials.from_service_account_info(
                    info, scopes=scopes
                )

            if creds:
                self.client = build("drive", "v3", credentials=creds)
                print(f"[GDrive] Google Drive service ready for folder: {cfg['folder_id']}")
        except Exception as e:
            print(f"[GDrive Warning] Failed to initialize Google Drive client: {e}")
            self.client = None

    def upload_via_webhook(
        self,
        file_path: str,
        file_name: str,
        mime_type: str = "audio/wav",
        target_folder_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Uploads audio file to Google Drive using Google Apps Script Webhook Relay.
        This writes directly under user's personal Drive quota without Service Account 403 errors.
        """
        cfg = _get_current_config()
        webhook_url = cfg["webhook_url"]
        if not webhook_url or "/exec" not in webhook_url:
            return None

        try:
            with open(file_path, "rb") as f:
                b64_content = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "folderId": target_folder_id or cfg["folder_id"],
                "filename": file_name,
                "mimeType": mime_type or "audio/wav",
                "base64": b64_content,
            }

            resp = requests.post(webhook_url, json=payload, timeout=15, allow_redirects=True)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if data.get("success"):
                        return {
                            "success": True,
                            "file_id": data.get("fileId"),
                            "file_name": file_name,
                            "folder_id": target_folder_id or cfg["folder_id"],
                            "web_view_link": data.get("url"),
                        }
                    else:
                        print(f"[GDrive Webhook Error] Apps Script error: {data.get('error')}")
                except Exception as json_err:
                    print(f"[GDrive Webhook Response] Raw text: {resp.text[:200]}")
            else:
                print(f"[GDrive Webhook Warning] HTTP {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            print(f"[GDrive Webhook Error] {e}")

        return None

    def upload_file(
        self,
        file_path: str,
        file_name: Optional[str] = None,
        mime_type: str = "audio/wav",
        target_folder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Uploads a local audio file to the Google Drive shared folder.
        """
        cfg = _get_current_config()
        folder = target_folder_id or cfg["folder_id"]
        path_obj = Path(file_path)
        name = file_name or path_obj.name

        if not path_obj.exists():
            return {
                "success": False,
                "error": f"Local file not found at {file_path}",
                "file_name": name,
            }

        # 1. Primary Method: Google Apps Script Webhook Relay (Bypasses Service Account 0GB personal quota limit)
        webhook_res = self.upload_via_webhook(str(path_obj), name, mime_type, folder)
        if webhook_res and webhook_res.get("success"):
            print(f"[GDrive Success] Uploaded '{name}' to Google Drive via Webhook.")
            return webhook_res

        # 2. Secondary Method: Google Drive v3 API client (Service Account or OAuth)
        if not self.client:
            self._init_client()

        if self.client:
            try:
                from googleapiclient.http import MediaFileUpload

                file_metadata = {
                    "name": name,
                    "parents": [folder],
                }

                media = MediaFileUpload(
                    str(path_obj),
                    mimetype=mime_type or "audio/wav",
                    resumable=True,
                )

                created_file = (
                    self.client.files()
                    .create(
                        body=file_metadata,
                        media_body=media,
                        fields="id, name, webViewLink, webContentLink, size",
                        supportsAllDrives=True,
                    )
                    .execute()
                )

                file_id = created_file.get("id")
                web_view_link = created_file.get("webViewLink")

                print(f"[GDrive Success] Uploaded '{name}' -> File ID: {file_id}")

                return {
                    "success": True,
                    "file_id": file_id,
                    "file_name": name,
                    "folder_id": folder,
                    "web_view_link": web_view_link,
                    "size": created_file.get("size"),
                }
            except Exception as e:
                err_str = str(e)
                # Catch quota exceeded specifically
                if "storageQuotaExceeded" in err_str:
                    err_msg = "Google Drive service account has 0GB storage on personal Gmail. Please use the Google Apps Script Webhook URL."
                else:
                    err_msg = err_str
                print(f"[GDrive API Warning] {err_msg}")
                return {
                    "success": False,
                    "error": err_msg,
                    "file_name": name,
                    "folder_id": folder,
                }

        return {
            "success": False,
            "error": "Google Drive authentication not configured. Audio saved locally.",
            "file_name": name,
            "folder_id": folder,
        }


gdrive_service = GoogleDriveService()
