"""SQLAlchemy engine / session helpers. Creates all tables on init."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import load_settings
from app.db.models import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    settings = load_settings()
    url = (settings.get("DATABASE_URL") or "").strip()
    if url:
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return url
    user = settings.get("DB_USER") or "clinIQ_user"
    password = settings.get("DB_PASSWORD") or "password"
    host = settings.get("DB_HOST") or "127.0.0.1"
    port = settings.get("DB_PORT") or "5432"
    name = settings.get("DB_NAME") or "cliniIQ"
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def get_engine(database_url: Optional[str] = None) -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = database_url or get_database_url()
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        _engine = create_engine(url, pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(
            bind=_engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )
    return _engine


def get_session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def init_db(database_url: Optional[str] = None) -> dict:
    """
    One-click DB bootstrap: create all tables if they do not exist.
    Safe to call on every app startup / deploy.
    """
    global _engine, _SessionLocal
    if database_url:
        _engine = None
        _SessionLocal = None
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {
        "enabled": True,
        "schema_applied": True,
        "orm": "sqlalchemy",
        "tables": sorted(Base.metadata.tables.keys()),
        "database": str(engine.url).split("@")[-1],
    }
