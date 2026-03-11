"""
Proof of concept: Config-driven multi-variant collection loop with HITL gate.

All HITL policy lives in hitl_config.toml:
  - autonomous = true/false          (master switch for the whole run)
  - [hitl.T14_stretch] enabled = ... (per-tool on/off)
  - [[hitl.T14_stretch.variants]]    (N variants; change count here, not in code)

The graph mechanics:
  1. ReAct loop (llm → action → llm) runs with zero interrupts.
  2. When the LLM produces no tool_calls it exits → hits hitl node.
  3. HITL Phase 1: variants incomplete → inject next instruction → back to llm.
  4. HITL Phase 2: all variants present → interrupt() for human review.
  5. Feedback: clear variants → back to llm (agent re-produces all variants).
  6. Affirmative: proceed.

Run:
    python scripts/hitl_variant_loop_poc.py
"""

from __future__ import annotations

import operator
import tomllib
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt


# ── Config loading ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "hitl_config.toml"


def load_hitl_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return raw


_CFG = load_hitl_config()


def tool_cfg(tool_id: str) -> dict:
    """Return the HITL config for a specific tool. Defaults to disabled."""
    return _CFG.get("hitl", {}).get(tool_id, {"enabled": False})


def is_enabled(tool_id: str) -> bool:
    if _CFG.get("autonomous", False):
        return False
    return tool_cfg(tool_id).get("enabled", False)


# ── Mock tool ──────────────────────────────────────────────────────────────────

@tool
def stretch_image(image_path: str, output_suffix: str, stretch_amount: float,
                  highlight_protection: float = 1.0) -> dict:
    """Apply a non-linear stretch to the image. Returns the output file path."""
    output_path = f"/tmp/stretched_{output_suffix}.fit"
    print(f"    [tool] stretch_image  suffix={output_suffix!r}  "
          f"amount={stretch_amount}  hp={highlight_protection}")
    return {
        "stretched_image_path": output_path,
        "output_suffix": output_suffix,
    }


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only
    stretch_variants: list                   # replaced wholesale on each write
    current_image: str
    image_state: dict                        # populated by analyze_image (T20) — SNR,
                                             # noise floor, clipping %, etc. The agent
                                             # reads this when choosing variant params.


# ── Generic variant collection helpers ────────────────────────────────────────

def _next_instruction(tool_name: str, image_path: str,
                      variants_spec: list, collected: list,
                      image_state: dict | None = None) -> str | None:
    """
    Return a goal-oriented instruction for the next missing variant.

    The agent receives the variant's conceptual goal and current image state,
    then chooses actual parameter values itself. This is intentional: the
    right stretch_amount for 'gentle' depends on the current SNR, noise floor,
    target type, and prior run preferences — not a hardcoded number.
    """
    existing = {v["output_suffix"] for v in collected}
    for spec in variants_spec:
        if spec["output_suffix"] not in existing:
            state_summary = ""
            if image_state:
                state_summary = (
                    f"\n\nCurrent image state:\n"
                    + "\n".join(f"  {k}: {v}" for k, v in image_state.items())
                )
            return (
                f"Produce the '{spec['output_suffix']}' variant using {tool_name}.\n"
                f"Set output_suffix='{spec['output_suffix']}' and "
                f"image_path={image_path!r}.\n\n"
                f"Goal: {spec['description'].strip()}"
                f"{state_summary}"
            )
    return None


# ── HITL helpers ───────────────────────────────────────────────────────────────

_AFFIRMATIVE = {
    "ok", "looks good", "continue", "proceed", "yes",
    "approve", "done", "good", "perfect", "great",
}


def _is_affirmative(response: str) -> bool:
    return response.strip().lower() in _AFFIRMATIVE


# ── Nodes ──────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an astrophotography processing agent. "
    "When given an instruction to call a tool, call it with exactly the parameters specified. "
    "After the tool call, if you have no further instructions, stop."
)

_model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools([stretch_image])


def llm_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=_SYSTEM)] + state["messages"]
    response = _model.invoke(messages)
    if response.tool_calls:
        names = [
            f"{tc['name']}({tc['args'].get('output_suffix', '')})"
            for tc in response.tool_calls
        ]
        print(f"  [llm]    → tool_calls: {names}")
    else:
        print("  [llm]    → no tool_calls — exiting ReAct loop → hitl")
    return {"messages": [response]}


def action_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_results = []
    new_variants = list(state.get("stretch_variants", []))

    for tc in last.tool_calls:
        if tc["name"] == "stretch_image":
            result = stretch_image.invoke(tc["args"])
            tool_results.append(ToolMessage(
                tool_call_id=tc["id"],
                name=tc["name"],
                content=str(result),
            ))
            new_variants.append({
                "output_suffix": tc["args"]["output_suffix"],
                "path": result["stretched_image_path"],
            })

    print(f"  [action] variants so far: {[v['output_suffix'] for v in new_variants]}")
    return {
        "messages": tool_results,
        "stretch_variants": new_variants,
    }


def hitl_stretch_node(state: AgentState) -> dict:
    """
    HITL node for T14 stretch. All policy comes from hitl_config.toml.

    Phase 1 — Collection: if variants are incomplete, inject the next
      instruction and route back to the llm. The ReAct loop restarts.
    Phase 2 — Review: all variants present, fire interrupt().
    """
    cfg = tool_cfg("T14_stretch")
    variants_spec = cfg.get("variants", [])
    collected = state.get("stretch_variants", [])

    # ── HITL disabled in config ────────────────────────────────────────────────
    if not is_enabled("T14_stretch"):
        # Pick the first variant as default (or moderate if present)
        moderate = next(
            (v for v in collected if v["output_suffix"] == "moderate"), None
        )
        default = moderate or (collected[0] if collected else None)
        print("  [hitl]   HITL disabled for T14 — skipping interrupt.")
        return {"current_image": default["path"]} if default else {}

    # ── Phase 1: Collection ────────────────────────────────────────────────────
    image_path = state.get("current_image", "/tmp/master_light.fit")
    instruction = _next_instruction(
        cfg["tool_name"], image_path, variants_spec, collected,
        image_state=state.get("image_state"),
    )
    if instruction:
        n_have = len(collected)
        n_need = len(variants_spec)
        print(f"  [hitl]   Phase 1 — {n_have}/{n_need} variants. Injecting instruction.")
        print(f"           → {instruction}")
        return {"messages": [HumanMessage(content=instruction)]}
        # route_hitl sees len(collected) < N → returns "llm"

    # ── Phase 2: Review ────────────────────────────────────────────────────────
    print(f"  [hitl]   Phase 2 — all {len(variants_spec)} variants ready. "
          f"Firing interrupt().")
    response = interrupt({
        "type": cfg["type"],
        "title": cfg["title"],
        "variants": [
            {"suffix": v["output_suffix"], "path": v["path"]} for v in collected
        ],
    })
    print(f"  [hitl]   Human responded: {response!r}")

    if _is_affirmative(str(response)):
        print("  [hitl]   Affirmative — proceeding.")
        return {}

    # Feedback: clear variants so Phase 1 restarts with updated messages.
    print("  [hitl]   Feedback — clearing variants, returning to agent.")
    return {
        "stretch_variants": [],
        "messages": [HumanMessage(content=str(response))],
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def route_llm(state: AgentState) -> str:
    """Stay in ReAct loop if tool_calls present, else exit to HITL."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "action"
    return "hitl_stretch"


def route_hitl(state: AgentState) -> str:
    """Back to llm if variants incomplete, else done."""
    n_need = len(tool_cfg("T14_stretch").get("variants", []))
    n_have = len(state.get("stretch_variants", []))
    # HITL disabled or autonomous: variants may be 0, just proceed
    if not is_enabled("T14_stretch"):
        return END
    if n_have < n_need:
        return "llm"
    return END


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("action", action_node)
    builder.add_node("hitl_stretch", hitl_stretch_node)

    builder.set_entry_point("llm")
    builder.add_conditional_edges(
        "llm", route_llm,
        {"action": "action", "hitl_stretch": "hitl_stretch"},
    )
    builder.add_edge("action", "llm")           # free loop — no interrupt here
    builder.add_conditional_edges(
        "hitl_stretch", route_hitl,
        {"llm": "llm", END: END},
    )

    return builder.compile(checkpointer=checkpointer)


# ── Driver ─────────────────────────────────────────────────────────────────────

def run():
    config = {"configurable": {"thread_id": "poc-stretch-001"}}

    print(f"\nLoaded config from: {_CONFIG_PATH}")
    cfg = tool_cfg("T14_stretch")
    print(f"T14 HITL enabled: {is_enabled('T14_stretch')}")
    print(f"T14 variants:     {[v['output_suffix'] for v in cfg.get('variants', [])]}")
    print(f"autonomous:       {_CFG.get('autonomous', False)}")

    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = build_graph(checkpointer)

        # image_state is populated by T20 analyze_image in the real pipeline.
        # The agent reads this when deciding what "gentle" means numerically
        # for this specific stack.
        mock_image_state = {
            "median_pixel":            0.0031,
            "background_noise":        0.00082,
            "estimated_snr_db":        21.6,
            "clipped_shadows_pct":     0.0,
            "clipped_highlights_pct":  0.0,
            "target_type":             "emission_nebula",
            "integration_hours":       2.1,
        }

        initial_state = {
            "messages": [HumanMessage(
                content="Please produce the required stretch variants for /tmp/master_light.fit."
            )],
            "stretch_variants": [],
            "current_image": "/tmp/master_light.fit",
            "image_state":   mock_image_state,
        }

        # ── Run 1 ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("RUN 1: Initial")
        print("=" * 60)

        for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                iv = chunk["__interrupt__"][0].value
                print(f"\n>>> HITL INTERRUPT:")
                print(f"    type:     {iv['type']}")
                print(f"    title:    {iv['title']}")
                print(f"    variants: {[v['suffix'] for v in iv['variants']]}")

        # ── Feedback ──────────────────────────────────────────────────────────
        feedback = "moderate is closest but the outer shell is too dim — push further"
        print(f"\n>>> Human feedback: {feedback!r}")
        print("\n" + "=" * 60)
        print("RUN 2: Resume with feedback")
        print("=" * 60)

        for chunk in graph.stream(
            Command(resume=feedback), config=config, stream_mode="updates"
        ):
            if "__interrupt__" in chunk:
                iv = chunk["__interrupt__"][0].value
                print(f"\n>>> HITL INTERRUPT (second pass):")
                print(f"    variants: {[v['suffix'] for v in iv['variants']]}")

        # ── Approval ──────────────────────────────────────────────────────────
        approval = "looks good"
        print(f"\n>>> Human approval: {approval!r}")
        print("\n" + "=" * 60)
        print("RUN 3: Resume with approval")
        print("=" * 60)

        for chunk in graph.stream(
            Command(resume=approval), config=config, stream_mode="updates"
        ):
            pass

        final = graph.get_state(config)
        suffixes = [v["output_suffix"] for v in final.values.get("stretch_variants", [])]
        print("\n" + "=" * 60)
        print(f"DONE. Final variants in state: {suffixes}")
        print("=" * 60)


if __name__ == "__main__":
    run()
