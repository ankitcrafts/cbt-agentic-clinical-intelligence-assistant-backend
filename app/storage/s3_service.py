"""S3 uploads for visit PDFs / conversation JSON.

If S3_BUCKET is unset, uploads are skipped and local disk remains the source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.core.config import load_settings


class S3Service:
    def __init__(self):
        settings = load_settings()
        self.bucket = settings["S3_BUCKET"]
        self.prefix = (settings["S3_PREFIX"] or "clinical-visits").strip("/")
        self.region = settings["AWS_REGION"] or "ap-south-1"
        self.access_key = settings["AWS_ACCESS_KEY_ID"]
        self.secret_key = settings["AWS_SECRET_ACCESS_KEY"]
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)

    def _get_client(self):
        if self._client is not None:
            return self._client
        import boto3

        kwargs = {"region_name": self.region}
        if self.access_key and self.secret_key:
            kwargs["aws_access_key_id"] = self.access_key
            kwargs["aws_secret_access_key"] = self.secret_key
        self._client = boto3.client("s3", **kwargs)
        return self._client

    def upload_file(
        self,
        local_path: str,
        *,
        visit_id: str,
        artifact_type: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        path = Path(local_path)
        if not path.exists():
            return {"enabled": True, "uploaded": False, "error": f"missing file: {local_path}"}

        key = f"{self.prefix}/{visit_id}/{artifact_type}/{path.name}"
        client = self._get_client()
        extra_args = None
        if path.suffix.lower() == ".pdf":
            extra_args = {"ContentType": "application/pdf"}
        elif path.suffix.lower() == ".json":
            extra_args = {"ContentType": "application/json"}
        if extra_args:
            client.upload_file(str(path), self.bucket, key, ExtraArgs=extra_args)
        else:
            client.upload_file(str(path), self.bucket, key)
        url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"
        return {
            "enabled": True,
            "uploaded": True,
            "s3_key": key,
            "s3_url": url,
            "bucket": self.bucket,
        }
