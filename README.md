# AstroAgent

Agentic astrophotography post-processing. Point it at a dataset, tell it the target
and sky conditions, and let it run. HITL gates pause for human review at subjective
decision points (stretch, star restoration, etc.).

## Dependencies

External binaries — install before running:

| Tool | Install |
|---|---|
| [Siril](https://siril.org) | `brew install --cask siril` or download .dmg |
| [GraXpert](https://www.graxpert.com) | Download .dmg |
| [StarNet2](https://www.starnetastro.com/download/) | Download binary, see `.env.example` for setup |
| ExifTool | `brew install exiftool` |

Python environment:

```bash
uv sync
```

## Configuration

```bash
cp .env.example .env
```

Edit `.env` for binary paths and LLM model. API keys go in your shell environment
(not in `.env`):

```bash
# ~/.zshrc
export TOGETHER_API_KEY=...       # default provider
export LANGCHAIN_API_KEY=...      # optional — LangSmith tracing
export LANGCHAIN_TRACING_V2=true  # optional
```

Edit `equipment.toml` with your camera and telescope specs.

## Dataset Layout

```
my-dataset/
├── lights/     # or: light/
├── darks/      # or: dark/
├── flats/      # or: flat/
└── bias/       # or: biases/ or bias_frames/
```

Files are never copied — only paths are read.

## Running

### Gradio (recommended)

```bash
uv run python -m astro_agent.gradio_app
```

The app opens with a Processing tab (chat + image gallery + activity log),
Equipment, HITL Config, and Model & Limits tabs. Enter the dataset path and
target name, then click **Start Processing**. The agent streams tool calls
in the activity log while you watch progress. HITL gates pause and show
images in the gallery — chat with the agent or click **Approve** to continue.

Resume any session by entering the thread ID in the **Resume Session** section.

### CLI

```bash
astro-agent process /path/to/dataset --target "M42 Orion Nebula" --bortle 5
```

Options:
- `--sqm 20.8` — SQM-L sky quality reading (more precise than Bortle)
- `--notes "L-eNhance filter, gain 100"` — injected into agent context every step
- `--resume run-m42-20260311-120000` — resume from checkpoint
- `--autonomous` — skip all HITL gates (fully automated)
- `--db checkpoints.db` — custom checkpoint database path

At a HITL gate: type `a` to approve, or type feedback to continue the conversation.

## Logs

Each session writes a log file to `~/.astroagent/logs/<thread_id>.log`. To follow a
running session in real time:

```bash
tail -f ~/.astroagent/logs/run-*.log
```

The log captures graph-level events (phase routing, agent decisions, HITL state) that
aren't surfaced in the Gradio UI. Useful for debugging after the fact.
LangSmith tracing (`LANGCHAIN_TRACING_V2=true`) captures the full LLM reasoning trace.

## LangGraph Studio

The graph is exposed at `astro_agent/graph/graph.py:graph` and registered in
`langgraph.json`. Open with the LangGraph Studio desktop app for visual graph
inspection and step-through debugging.
