"""Database session and engine helpers."""

import os

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.models.research_models import Base


def get_engine(db_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine.

    Reads DATABASE_URL from environment if db_url is not provided.
    Defaults to a local SQLite file.
    """
    if db_url is None:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///factor_research.db")
    return create_engine(db_url)


def get_session(engine: Engine) -> Session:
    """Create a new database session."""
    session_factory = sessionmaker(bind=engine)
    return session_factory()


def init_db(engine: Engine) -> None:
    """Create all tables. Used for Phase 1 standalone; Alembic for fin-arb."""
    Base.metadata.create_all(engine)
