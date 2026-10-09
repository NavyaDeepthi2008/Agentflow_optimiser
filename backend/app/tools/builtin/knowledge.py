"""Knowledge-base tools: ``kb_search`` and ``doc_read`` over the fictional corpus."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.builtin.base import MockTool
from app.tools.builtin.corpus import DOCUMENTS
from app.tools.errors import ToolNotFoundError
from app.tools.spec import CacheScope, Freshness, ToolEffect, ToolSpec

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by does do for from how in is it of on or the to what when "
    "which who why with".split()
)
SNIPPET_CHARS = 140


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


class KbSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=200, description="What to search for.")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of results.")


class DocReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str = Field(pattern=r"^kb-\d{3}$", description="Document id, e.g. 'kb-002'.")


class KbSearch(MockTool):
    spec = ToolSpec(
        name="kb_search",
        description=(
            "Search the Nimbus Labs knowledge base. Returns matching document ids, titles "
            "and short snippets. Use doc_read to get a full document."
        ),
        args_model=KbSearchArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.TTL,
        ttl_seconds=3600,
        max_ttl_seconds=86400,
        cache_scope=CacheScope.SESSION,
        normalize_whitespace_args=("query",),
        casefold_args=("query",),  # the search itself ignores case and spacing
        retryable=True,
        timeout_seconds=5.0,
    )

    def run(self, args: KbSearchArgs) -> dict[str, Any]:
        query_terms = set(_terms(args.query))
        scored: list[tuple[int, str]] = []
        for doc_id, doc in DOCUMENTS.items():
            title_terms = _terms(doc["title"])
            body_terms = _terms(doc["body"])
            score = sum(2 for t in title_terms if t in query_terms) + sum(
                1 for t in body_terms if t in query_terms
            )
            if score > 0:
                scored.append((score, doc_id))
        # Highest score first; ties broken by document id so results are deterministic.
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = [
            {
                "doc_id": doc_id,
                "title": DOCUMENTS[doc_id]["title"],
                "snippet": DOCUMENTS[doc_id]["body"][:SNIPPET_CHARS],
                "score": score,
            }
            for score, doc_id in scored[: args.limit]
        ]
        return {"results": results}


class DocRead(MockTool):
    spec = ToolSpec(
        name="doc_read",
        description="Read the full text of one Nimbus Labs knowledge-base document by id.",
        args_model=DocReadArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.TTL,
        ttl_seconds=3600,
        max_ttl_seconds=86400,
        cache_scope=CacheScope.SESSION,
        resource_tags=("doc:{doc_id}",),
        retryable=True,
        timeout_seconds=5.0,
    )

    def run(self, args: DocReadArgs) -> dict[str, Any]:
        doc = DOCUMENTS.get(args.doc_id)
        if doc is None:
            raise ToolNotFoundError(f"document {args.doc_id} does not exist")
        return {"doc_id": args.doc_id, "title": doc["title"], "body": doc["body"]}
