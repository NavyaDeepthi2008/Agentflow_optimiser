"""Alembic environment.

The database URL comes from the Alembic config (set by ``app.storage.db``), or
from ``DATABASE_URL`` when Alembic is run from the command line. If a caller
passes an open connection in ``config.attributes["connection"]``, it is used
directly, so migrations run with the app's own engine settings.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import Connection

from app.storage.db import make_engine
from app.storage.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return (
        config.get_main_option("sqlalchemy.url")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///./agentflow.db"
    )


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # lets future migrations alter SQLite tables safely
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(
        url=_url(), target_metadata=target_metadata, literal_binds=True, render_as_batch=True
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = make_engine(_url())
    try:
        with engine.connect() as conn:
            _run(conn)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
