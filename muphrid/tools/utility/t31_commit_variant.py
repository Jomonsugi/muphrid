"""
T31 — commit_variant

Autonomous-mode counterpart of HITL variant approval. The agent uses this
tool to lock in one variant from state.variant_pool as its working image
and clear the rest. Mirrors what promote_variant does on human approval,
but is initiated by an explicit agent tool call rather than a human click
in Gradio.

Why it exists:
  In autonomous mode the agent often produces several variants of a step
  (e.g. three gradient-removal passes with different parameters) and needs
  a way to declare "this one is what I'm carrying forward". Without an
  explicit commit, there is no signal — paths.current_image just tracks the
  most recent tool output, which is NOT the same as "the chosen one". This
  tool gives the agent that signal, leveraging LangGraph state as the
  shared communication channel between the agent's intent and the rest of
  the pipeline.

State side effects (via build_variant_promotion_update):
  - paths.current_image := variant.file_path
  - variant_pool        := []   (other variants are dropped from the pool;
                                  their files remain on disk)
  - visual_context      += phase_carry entry for the committed variant, so
                            the VLM (when enabled) keeps the chosen image
                            visible after this gate as the agent moves on

This is a UTILITY tool — available in every phase.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState

logger = logging.getLogger(__name__)


class CommitVariantInput(BaseModel):
    """Input schema for commit_variant."""
    variant_id: str = Field(
        description=(
            "The id of the variant to commit, e.g. 'T09_v3'. Variant ids are "
            "issued by variant_snapshot when HITL-mapped tools execute, in the "
            "format '<tool_short_code>_v<n>'. If you don't know the id, run "
            "the tool again or check your prior tool results."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description="Optional reason for choosing this variant — preserved in logs.",
    )


@tool(args_schema=CommitVariantInput)
def commit_variant(
    variant_id: str,
    rationale: str | None = None,
    state: Annotated[AstroState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    Commit one variant from the active pool to current_image and clear the pool.

    Use this when you have produced multiple variants of a step (e.g. several
    gradient removal passes with different parameters) and want to lock in one
    as your working image before moving on. The other variants are dropped
    from the pool but their files remain on disk for reference. The chosen
    variant becomes the input to the next tool call.

    Autonomous-mode equivalent of HITL variant approval. In HITL mode the
    human picks the variant via Gradio; in autonomous mode you pick it
    yourself with this tool.
    """
    # Lazy import: registry → tools imports happen at module load, and nodes
    # imports registry, so a top-level `from muphrid.graph.nodes import ...`
    # creates a cycle. Calling at runtime avoids it.
    from muphrid.graph.nodes import build_variant_promotion_update

    state = state or {}
    pool = list(state.get("variant_pool", []) or [])

    result = build_variant_promotion_update(state, variant_id)
    if result is None:
        valid_ids = [v.get("id", "?") for v in pool]
        error_payload = {
            "status": "error",
            "message": (
                f"Variant '{variant_id}' not found in the current variant_pool. "
                f"Valid ids: {valid_ids if valid_ids else '(pool is empty)'}."
            ),
            "valid_ids": valid_ids,
        }
        logger.warning(
            f"commit_variant: id={variant_id!r} not in pool "
            f"(valid={valid_ids})"
        )
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps(error_payload),
                tool_call_id=tool_call_id,
            )],
        })

    variant, update = result
    success_payload = {
        "status": "committed",
        "variant_id": variant["id"],
        "label": variant.get("label"),
        "file_path": variant["file_path"],
        "rationale": rationale,
        "dropped_variants": [
            v.get("id") for v in pool if v.get("id") != variant["id"]
        ],
    }
    update["messages"] = [ToolMessage(
        content=json.dumps(success_payload),
        tool_call_id=tool_call_id,
    )]

    logger.info(
        f"commit_variant: {variant['id']} → current_image; "
        f"dropped {len(pool) - 1} other variant(s)"
        + (f" (rationale: {rationale})" if rationale else "")
    )
    return Command(update=update)
