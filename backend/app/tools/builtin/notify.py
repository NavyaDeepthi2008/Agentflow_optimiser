"""Mock notification tool (side effect): writes messages to an in-memory outbox.

Nothing is actually sent anywhere. Every successful execution appends to
``world.outbox`` so tests can prove a requested notification was never skipped.

If the caller passes an ``idempotency_key`` that was already used, the tool
returns the earlier message instead of sending a duplicate. Note that the tool
*still executes* in that case (it is invoked and counted); it just recognises
the repeat itself. That is the safe way to make retries of side effects harmless.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.builtin.base import MockTool
from app.tools.spec import Freshness, ToolEffect, ToolSpec


class SendNotificationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipient: str = Field(
        pattern=r"^(team|user):[a-z0-9_-]{1,32}$",
        description="Recipient, e.g. 'team:billing' or 'user:asha'.",
    )
    message: str = Field(min_length=1, max_length=500, description="Message text.")
    idempotency_key: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{8,64}$",
        description="Optional key; repeating a key returns the original message instead of "
        "sending again.",
    )


class SendNotification(MockTool):
    spec = ToolSpec(
        name="send_notification",
        description="Send a notification message to a team or user. This performs an action.",
        args_model=SendNotificationArgs,
        effect=ToolEffect.SIDE_EFFECT,
        freshness=Freshness.LIVE,
        invalidates=(),  # sending a message changes no cached data
        idempotency_arg="idempotency_key",
        # Retried only when the request carries an idempotency key (gateway rule).
        retryable=True,
        timeout_seconds=5.0,
    )

    def run(self, args: SendNotificationArgs) -> dict[str, Any]:
        world = self.world
        if args.idempotency_key and args.idempotency_key in world.idempotency_index:
            existing = world.outbox[world.idempotency_index[args.idempotency_key]]
            return {**existing, "duplicate_suppressed": True}
        entry = {
            "message_id": f"MSG-{len(world.outbox) + 1:04d}",
            "recipient": args.recipient,
            "message": args.message,
            "sent_at": world.clock.utcnow().isoformat(),
        }
        world.outbox.append(entry)
        if args.idempotency_key:
            world.idempotency_index[args.idempotency_key] = len(world.outbox) - 1
        return {**entry, "duplicate_suppressed": False}
