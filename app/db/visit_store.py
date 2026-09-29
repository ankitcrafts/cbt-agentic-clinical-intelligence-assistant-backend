"""PostgreSQL visit / conversation persistence via SQLAlchemy.

Tables are auto-created on ensure_schema() / app startup (one-click deploy).
If DB is unreachable, callers should fall back to in-memory.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import delete, select

from app.db.models import ConversationTurn, Visit, VisitArtifact
from app.db.session import get_database_url, get_session, init_db


class VisitStore:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = (database_url or get_database_url()).strip()
        self._ensured = False

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    def ensure_schema(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "message": "DATABASE_URL not set"}
        result = init_db(self.database_url)
        self._ensured = True
        return result

    def _ensure(self) -> None:
        if not self._ensured:
            self.ensure_schema()

    def save_visit(
        self,
        result: dict[str, Any],
        *,
        conversation_round: int = 0,
        replace_turns: bool = True,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        self._ensure()

        visit_id = result.get("visit_id")
        if not visit_id:
            raise ValueError("visit_id required to persist visit")

        # JSON-safe payload for JSONB
        payload = json.loads(json.dumps(result, default=str))

        session = get_session()
        try:
            visit = session.get(Visit, visit_id)
            if visit is None:
                visit = Visit(visit_id=visit_id)
                session.add(visit)

            visit.status = result.get("status")
            visit.follow_up_iteration = int(result.get("follow_up_iteration") or 0)
            visit.confidence = result.get("confidence")
            visit.previous_confidence = result.get("previous_confidence")
            visit.source_file = result.get("source_file")
            visit.payload = payload

            if replace_turns:
                session.execute(
                    delete(ConversationTurn).where(ConversationTurn.visit_id == visit_id)
                )
                turns = (
                    result.get("doctor_patient_conversation")
                    or result.get("conversation")
                    or []
                )
                for index, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    text = str(turn.get("text") or "").strip()
                    if not text:
                        continue
                    round_no = int(result.get("follow_up_iteration") or conversation_round)
                    if turn.get("round") is not None:
                        round_no = int(turn["round"])
                    session.add(
                        ConversationTurn(
                            visit_id=visit_id,
                            round=round_no,
                            turn_index=index,
                            role=turn.get("role"),
                            speaker=turn.get("speaker"),
                            text=text,
                        )
                    )

            session.commit()
            return {"enabled": True, "saved": True, "visit_id": visit_id, "orm": "sqlalchemy"}
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_visit(self, visit_id: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        self._ensure()
        session = get_session()
        try:
            visit = session.get(Visit, visit_id)
            if not visit:
                return None
            payload = visit.payload or {}
            if isinstance(payload, str):
                return json.loads(payload)
            return dict(payload)
        finally:
            session.close()

    def save_artifact(
        self,
        visit_id: str,
        artifact_type: str,
        *,
        local_path: Optional[str] = None,
        s3_key: Optional[str] = None,
        s3_url: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        self._ensure()
        session = get_session()
        try:
            # Ensure parent visit row exists
            if session.get(Visit, visit_id) is None:
                session.add(Visit(visit_id=visit_id, payload={}))
            session.add(
                VisitArtifact(
                    visit_id=visit_id,
                    artifact_type=artifact_type,
                    local_path=local_path,
                    s3_key=s3_key,
                    s3_url=s3_url,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_conversation(self, visit_id: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        self._ensure()
        session = get_session()
        try:
            rows = session.scalars(
                select(ConversationTurn)
                .where(ConversationTurn.visit_id == visit_id)
                .order_by(ConversationTurn.turn_index.asc())
            ).all()
            return [
                {
                    "round": row.round,
                    "turn_index": row.turn_index,
                    "role": row.role,
                    "speaker": row.speaker,
                    "text": row.text,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        finally:
            session.close()
