"""Database engine, sessions and migrations.

The schema is created ONLY by Alembic migrations (``upgrade_to_head``), never by
``create_all``, so the database a user runs is exactly what the migrations
define. A test checks that the migrations and the models agree.

SQLite settings applied on every connection:
* ``foreign_keys=ON``: SQLite ignores foreign keys unless asked.
* ``journal_mode=WAL``: readers are not blocked while a run is writing.
* ``busy_timeout``: wait up to 5 s for a lock instead of failing immediately.

Only SQLAlchemy Core/ORM is used, so switching to PostgreSQL later is a change
of ``DATABASE_URL`` plus testing, not a rewrite.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    kwargs: dict = {}
    if parsed.get_backend_name() == "sqlite":
        # The API serves requests and runs agents from different threads.
        kwargs["connect_args"] = {"check_same_thread": False}
        if parsed.database not in (None, "", ":memory:"):
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)

    if parsed.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            if parsed.database not in (None, "", ":memory:"):
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", url)
    return config


def upgrade_to_head(url: str) -> None:
    command.upgrade(alembic_config(url), "head")


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine = make_engine(url)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def migrate(self) -> None:
        """Bring the schema up to date using this database's own engine."""
        config = alembic_config(self.url)
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    def session(self) -> Session:
        return self._sessions()

    def dispose(self) -> None:
        self.engine.dispose()
