"""SQLAlchemy models for clinical visit persistence."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Visit(Base):
    __tablename__ = "visits"

    visit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    follow_up_iteration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    previous_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_file: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    turns: Mapped[List["ConversationTurn"]] = relationship(
        back_populates="visit",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.turn_index",
    )
    artifacts: Mapped[List["VisitArtifact"]] = relationship(
        back_populates="visit",
        cascade="all, delete-orphan",
    )


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visit_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("visits.visit_id", ondelete="CASCADE"), nullable=False
    )
    round: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    role: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    speaker: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    visit: Mapped["Visit"] = relationship(back_populates="turns")


class VisitArtifact(Base):
    __tablename__ = "visit_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    visit_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("visits.visit_id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    local_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    s3_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    s3_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    visit: Mapped["Visit"] = relationship(back_populates="artifacts")
