# AstroAgent

LLM-powered autonomous astrophotography post-processing. Point it at a dataset of
raw frames, tell it the target and sky conditions, and it handles calibration,
stacking, stretching, and enhancement — pausing at key decision points (stretch,
curves, star restoration) for your review.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/astro-agent.git
cd astro-agent
pip install uv          # if you don't have uv
uv sync

# 2. Configure
cp .env.example .env    # then edit — see "Setup" below

# 3. Launch
uv run python -m astro_agent
```

Enter your dataset path and target name in the UI, click **Start Processing**.

## Setup

### Python

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/) for dependency management.

### System binaries

Install these before running. AstroAgent checks for them at startup and will
tell you exactly what's missing and how to fix it.

| Tool | Install | Notes |
|------|---------|-------|
| [Siril](https://siril.org) >= 1.4 | `brew install --cask siril` | Image processing engine |
| [GraXpert](https://www.graxpert.com) >= 3.0 | Download from [releases](https://github.com/Steffenhir/GraXpert/releases) | AI gradient extraction and denoising |
| [StarNet2](https://www.starnetastro.com/download/) | See setup below | Neural network star removal |
| ExifTool >= 12 | `brew install exiftool` | EXIF metadata reader |

#### StarNet2 setup (macOS)

StarNet2 is a standalone binary that needs code-signing on macOS:

```bash
# Download the MPS-accelerated build from https://www.starnetastro.com/download/
chmod +x /path/to/starnet2
xattr -d com.apple.quarantine /path/to/starnet2
codesign --force --sign - /path/to/starnet2
# If blocked: System Settings > Privacy & Security > Allow
```

### .env

Copy `.env.example` to `.env` and fill in binary paths:

```bash
SIRIL_BIN=/Applications/Siril.app/Contents/MacOS/siril-cli
GRAXPERT_BIN=/Applications/GraXpert.app/Contents/MacOS/GraXpert
STARNET_BIN=/path/to/starnet2
STARNET_WEIGHTS=/path/to/StarNet2_weights.pt
```

### API key

The default model is **Kimi K2.5** via Together AI. Set the key in your shell
profile or `.env`:

```bash
export TOGETHER_API_KEY=your-key-here
```

To use a different model, change `[model] default` in `processing.toml` and set
the corresponding key:

| Model | Provider | Key |
|-------|----------|-----|
| moonshotai/Kimi-K2.5 (default) | Together AI | `TOGETHER_API_KEY` |
| claude-sonnet-4-20250514 | Anthropic | `ANTHROPIC_API_KEY` |
| gpt-4o | OpenAI | `OPENAI_API_KEY` |

## Dataset layout

```
my-dataset/
  lights/    # or light/
  darks/     # or dark/
  flats/     # or flat/
  bias/      # or biases/, bias_frames/
```

Camera RAW (`.RAF`, `.CR2`, `.ARW`, etc.) or FITS files. Original files are never
modified — all outputs go to a `runs/<session-id>/` folder.

## Running

### Gradio (recommended)

```bash
uv run python -m astro_agent
```

The UI has four tabs:

- **Processing** — chat, image gallery, activity log. Enter dataset path and
  target, click Start Processing. HITL gates pause and show images in the
  gallery — chat with the agent or click Approve to continue.
- **Equipment** — override pixel size, sensor type, focal length (see below).
- **HITL Config** — toggle which tools pause for review, enable autonomous mode.
- **Model & Limits** — switch models, adjust safety limits.

Resume a previous session by entering the thread ID in the Resume section.

### CLI

```bash
astro-agent process /path/to/dataset --target "M42 Orion Nebula" --bortle 5
```

| Flag | Description |
|------|-------------|
| `--sqm 20.8` | SQM-L sky quality reading |
| `--notes "L-eNhance, gain 100"` | Context injected every step |
| `--resume run-m42-20260311-120000` | Resume from checkpoint |
| `--autonomous` | Skip all HITL gates |
| `--memory` | Enable long-term memory |
| `--db checkpoints.db` | Custom checkpoint DB path |

At a HITL gate, type `a` to approve or type feedback to continue the conversation.

## Configuration files

### processing.toml

Model selection, safety limits, per-phase tool limits, behavior flags, memory,
and tracing. Good defaults — most users don't need to change anything here
except possibly the model.

### equipment.toml

Camera and telescope specs that can't be read from file metadata.

- **FITS cameras** (ZWO, QHY, etc.): pixel size and focal length are in FITS
  headers. Leave `equipment.toml` commented out — nothing to configure.
- **DSLR/mirrorless RAW** (`.RAF`, `.CR2`): pixel size is NOT in EXIF. Uncomment
  and set `[camera] pixel_size_um`. X-Trans sensors also need `sensor_type = "xtrans"`.

### hitl_config.toml

Controls which pipeline steps pause for human review. Four gates are enabled
by default: gradient removal, stretch, curves, and star restoration. You can
toggle each tool independently or enable autonomous mode to skip all gates.

## Long-term memory (optional)

When enabled, the agent learns from HITL sessions — what worked, what failed,
what you preferred. Memory is off by default. Enable it in `processing.toml`
when the agent is producing quality results:

```toml
[memory]
enabled = true
embedding_provider = "ollama"       # or "fastembed"
embedding_model = "qwen3-embedding" # must match your provider
```

**Ollama** — external service, large models, best quality. Requires
[Ollama](https://ollama.com) installed and the model pulled
(`ollama pull qwen3-embedding`). AstroAgent auto-starts the service if needed.

**FastEmbed** — in-process ONNX inference, no external service. Lighter models,
downloaded automatically on first use. Good alternative if you don't want to
run Ollama.

The embedding model is a one-time choice. Changing it requires rebuilding the
vector index: set `rebuild_embeddings = true` in `processing.toml` (or pass
`--rebuild-embeddings` on the CLI), then remove the flag after one run.

## Troubleshooting

AstroAgent validates all dependencies at startup. If something is missing,
the error message tells you exactly what to install and how.

| Error | Fix |
|-------|-----|
| siril-cli not found | Install Siril >= 1.4: `brew install --cask siril` |
| GraXpert binary not found | Download from [GraXpert releases](https://github.com/Steffenhir/GraXpert/releases), set `GRAXPERT_BIN` in `.env` |
| StarNet binary not found | Download, code-sign (see setup above), set `STARNET_BIN` and `STARNET_WEIGHTS` in `.env` |
| StarNet not executable | `chmod +x /path/to/starnet2` |
| ExifTool not found | `brew install exiftool` |
| TOGETHER_API_KEY not set | `export TOGETHER_API_KEY=...` in your shell profile or `.env` |
| Pixel size could not be determined | Set `pixel_size_um` in `equipment.toml` (DSLR/mirrorless only) |
| Embedding model changed | Set `rebuild_embeddings = true` in `processing.toml`, run once, then remove the flag |

### Logs

Each processing run writes a `processing_log.md` to its run folder
(`runs/<session-id>/processing_log.md`). For deeper debugging, enable
[LangSmith](https://smith.langchain.com/) tracing in `processing.toml`.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
