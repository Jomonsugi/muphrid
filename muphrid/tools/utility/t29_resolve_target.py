"""
T29 — resolve_target

Resolve an astronomical target name to RA/DEC coordinates (J2000, decimal degrees)
via SIMBAD. Returns coordinates that are passed as target_coords / approximate_coords
to any tool that performs plate solving (T10 color_calibrate, T21 plate_solve).

This is a mandatory early step in every pipeline run. The target name is required
input — the pipeline cannot proceed to plate-dependent operations (T10, T21) without
known coordinates, since blind wide-field plate solving is unreliable without a
position hint, especially for manual lenses without EXIF focal length metadata.

Call this immediately after T01 ingest. Store ra/dec in agent state and pass to
T10 and T21.

Requires internet access (queries the CDS SIMBAD name resolver).
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.equipment import resolve_target_coords
from muphrid.graph.state import AstroState


class ResolveTargetInput(BaseModel):
    target_name: str | None = Field(
        default=None,
        description=(
            "Target name to resolve. If None, reads from session state. "
            "SIMBAD accepts catalog names ('M42', 'NGC 1976') and common names "
            "('Orion Nebula', 'Andromeda Galaxy'). Do NOT pass free-form strings "
            "like 'M42 Orion Nebula' — use just 'M42' or just 'Orion Nebula'."
        ),
    )


def _try_resolve(name: str):
    """Try SIMBAD with the given name, then with just the first token."""
    coords = resolve_target_coords(name)
    if coords is not None:
        return coords, name
    # Try first word only (e.g. "M42" from "M42 Orion Nebula")
    first_token = name.split()[0]
    if first_token != name:
        coords = resolve_target_coords(first_token)
        if coords is not None:
            return coords, first_token
    return None, name


@tool(args_schema=ResolveTargetInput)
def resolve_target(
    target_name: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Resolve the session target name to RA/DEC coordinates via SIMBAD.

    Reads the target name from session state (set at pipeline initialization),
    or accepts an explicit override. Stores the resolved coordinates in metadata
    for downstream plate solving tools (T10, T21).

    SIMBAD accepts catalog names ('M42', 'NGC 1976', 'IC 434') and common names
    ('Orion Nebula', 'Andromeda Galaxy'). Do NOT pass combined strings like
    'M42 Orion Nebula' — use just 'M42' or just 'Orion Nebula'.

    If resolution fails, the tool automatically retries with just the first word
    of the name before returning a failure result.
    """
    session_name = state["session"]["target_name"]
    query = target_name or session_name
    coords, resolved_via = _try_resolve(query)
    acq = state["dataset"].get("acquisition_meta") or {}
    if coords is None:
        failure_summary = {
            "status": "failed",
            "target_name": query,
            "error": (
                f"SIMBAD could not resolve '{query}'. "
                "Try a shorter catalog name ('M42', 'NGC 1976') or a common name "
                "('Orion Nebula'). Avoid combined strings like 'M42 Orion Nebula'."
            ),
        }
        return Command(update={
            "dataset": {
                **state["dataset"],
                "acquisition_meta": {**acq, "target_coords": None},
            },
            "messages": [ToolMessage(content=json.dumps(failure_summary, indent=2, default=str), tool_call_id=tool_call_id)],
        })

    summary = {
        "status": "resolved",
        "queried_as": query,
        "resolved_via": resolved_via,
        "ra": coords["ra"],
        "dec": coords["dec"],
    }
    return Command(update={
        "dataset": {
            **state["dataset"],
            "acquisition_meta": {
                **acq,
                "target_coords": {"ra": coords["ra"], "dec": coords["dec"]},
            },
        },
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
