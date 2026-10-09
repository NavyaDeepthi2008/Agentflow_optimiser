"""Alembic migrations create exactly the schema the models describe."""

from __future__ import annotations

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from app.storage.db import Database, alembic_config
from app.storage.models import Base

TABLES = {"runs", "llm_calls", "tool_calls", "trace_events"}


def test_upgrade_creates_all_tables(tmp_path):
    db = Database(f"sqlite:///{tmp_path}/m.db")
    db.migrate()
    assert TABLES <= set(inspect(db.engine).get_table_names())


def test_migrations_match_the_models(tmp_path):
    db = Database(f"sqlite:///{tmp_path}/m.db")
    db.migrate()
    with db.engine.connect() as conn:
        diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diffs == []


def test_upgrade_is_repeatable_and_downgrade_removes_tables(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    db = Database(url)
    db.migrate()
    db.migrate()  # second run is a no-op
    command.downgrade(alembic_config(url), "base")
    assert not TABLES & set(inspect(db.engine).get_table_names())
    db.migrate()
    assert TABLES <= set(inspect(db.engine).get_table_names())


def test_sqlite_pragmas(tmp_path):
    db = Database(f"sqlite:///{tmp_path}/m.db")
    with db.engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 5000


def test_parent_directories_are_created(tmp_path):
    db = Database(f"sqlite:///{tmp_path}/nested/dir/m.db")
    db.migrate()
    assert (tmp_path / "nested" / "dir" / "m.db").exists()
