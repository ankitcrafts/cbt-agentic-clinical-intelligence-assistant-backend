import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import DATABASE_URL, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB

logger = logging.getLogger(__name__)

Base = declarative_base()


def ensure_postgres_database():
    """Ensure the target PostgreSQL database exists before connecting."""
    if not DATABASE_URL.startswith("postgresql"):
        return

    admin_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
    try:
        temp_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with temp_engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{POSTGRES_DB}'")
            )
            exists = result.scalar()
            if not exists:
                conn.execute(text(f"CREATE DATABASE {POSTGRES_DB}"))
                logger.info(f"Created PostgreSQL database: {POSTGRES_DB}")
        temp_engine.dispose()
    except Exception as e:
        logger.warning(f"Could not auto-create database '{POSTGRES_DB}': {e}")


def create_db_engine():
    """Create database engine with fallback for seamless developer experience."""
    ensure_postgres_database()
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database: {POSTGRES_DB}")
        return engine
    except Exception as e:
        logger.error(f"PostgreSQL connection failed ({e}). Falling back to local SQLite database.")
        sqlite_url = "sqlite:///./cliniq_local.db"
        return create_engine(sqlite_url, connect_args={"check_same_thread": False})


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for yielding database sessions per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and seed default accounts."""
    import app.models.models  # Ensure models are imported for metadata registration
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
    
    # Seed default demonstration users if table is empty
    from app.models.models import User
    db = SessionLocal()
    try:
        count = db.query(User).count()
        if count == 0:
            demo_users = [
                User(
                    email="admin@cliniq.health",
                    full_name="Hospital Administrator",
                    role="ADMIN",
                    department="Hospital Operations",
                    designation="System Administrator",
                ),
                User(
                    email="doctor@cliniq.health",
                    full_name="Dr. Yuki Tanaka",
                    role="DOCTOR",
                    department="Cardiology",
                    specialization="Senior Cardiologist",
                    designation="Attending Physician",
                ),
                User(
                    email="patient@cliniq.health",
                    full_name="Marcus Delgado",
                    role="PATIENT",
                    phone_number="+1-555-0199",
                ),
            ]
            db.add_all(demo_users)
            db.commit()
            logger.info("Seeded initial demo accounts (admin, doctor, patient).")
    except Exception as e:
        logger.error(f"Error seeding demo accounts: {e}")
        db.rollback()
    finally:
        db.close()
