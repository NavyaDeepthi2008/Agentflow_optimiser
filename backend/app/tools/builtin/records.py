"""Mock record tools: ``record_read`` (read-only) and ``record_update`` (side effect).

``record_update`` declares that it invalidates ``record:{record_id}``, which is
the tag ``record_read`` depends on. Stage 3 uses this to delete stale cached
reads after a write. In Stage 1 every read simply executes live.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.builtin.base import MockTool
from app.tools.errors import ToolInputError, ToolNotFoundError
from app.tools.spec import CacheScope, Freshness, ToolEffect, ToolSpec

ALLOWED_STATUSES = ("open", "in_progress", "resolved", "closed")
RECORD_ID_PATTERN = r"^REC-\d{4}$"


class RecordReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(pattern=RECORD_ID_PATTERN, description="Record id, e.g. 'REC-1001'.")


class RecordUpdateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(pattern=RECORD_ID_PATTERN, description="Record id, e.g. 'REC-1001'.")
    field: Literal["status", "owner", "note"] = Field(description="Which field to change.")
    value: str = Field(
        min_length=1,
        max_length=200,
        description="New value. For status: open, in_progress, resolved or closed.",
    )


class RecordRead(MockTool):
    spec = ToolSpec(
        name="record_read",
        description="Read one ticket record (title, status, owner, note, version) by id.",
        args_model=RecordReadArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.TTL,
        ttl_seconds=300,
        max_ttl_seconds=900,
        # Conservative default: tickets can be changed by people outside this
        # agent, so a stored read is shared only within one run. Stage 7's
        # learning engine may widen this to the session, with evidence and
        # human approval.
        cache_scope=CacheScope.RUN,
        resource_tags=("record:{record_id}",),
        retryable=True,
        timeout_seconds=5.0,
    )

    def run(self, args: RecordReadArgs) -> dict[str, Any]:
        record = self.world.records.get(args.record_id)
        if record is None:
            raise ToolNotFoundError(f"record {args.record_id} does not exist")
        return copy.deepcopy(record)


class RecordUpdate(MockTool):
    spec = ToolSpec(
        name="record_update",
        description=(
            "Change one field (status, owner or note) of a ticket record. This modifies data."
        ),
        args_model=RecordUpdateArgs,
        effect=ToolEffect.SIDE_EFFECT,
        freshness=Freshness.LIVE,
        invalidates=("record:{record_id}",),
        timeout_seconds=5.0,
    )

    def run(self, args: RecordUpdateArgs) -> dict[str, Any]:
        record = self.world.records.get(args.record_id)
        if record is None:
            raise ToolNotFoundError(f"record {args.record_id} does not exist")
        if args.field == "status" and args.value not in ALLOWED_STATUSES:
            raise ToolInputError(f"status must be one of {', '.join(ALLOWED_STATUSES)}")
        record[args.field] = args.value
        record["version"] = int(record["version"]) + 1
        return copy.deepcopy(record)
