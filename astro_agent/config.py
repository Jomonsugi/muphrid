"""
Environment loading, external dependency verification, and processing profiles.

Call check_dependencies() once at startup before any tool is invoked.
The LLM is NOT instantiated here — that happens in graph/planner.py (Phase 7).
This module only validates that the right API key and binaries are present.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ── Errors ─────────────────────────────────────────────────────────────────────

class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


class DependencyError(RuntimeError):
    """Raised when a required external binary or library is missing or outdated."""


# ── Environment helpers ────────────────────────────────────────────────────────

def _require(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise ConfigError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example → .env and fill in the missing value."
        )
    return value


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# ── Settings ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str       # "together" | "anthropic" | "openai"
    llm_model: str
    llm_temperature: float
    together_api_key: str   # required when llm_provider == "together"
    anthropic_api_key: str  # required when llm_provider == "anthropic"

    # External binaries
    siril_bin: str
    graxpert_bin: str
    starnet_bin: str        # absolute path to the starnet2 executable
    starnet_weights: str    # absolute path to StarNet2_weights.pt

    # Camera sensor
    pixel_size_um: float | None  # optional override; None → use camera lookup table

    # LangSmith (optional tracing)
    langchain_tracing: bool
    langchain_api_key: str
    langchain_project: str

    # Long-term memory
    memory_enabled: bool
    memory_db_path: str
    memory_embedding_model: str


def load_settings() -> Settings:
    provider = _optional("LLM_PROVIDER", "together").lower()

    if provider == "together":
        together_key = _require("TOGETHER_API_KEY")
        anthropic_key = _optional("ANTHROPIC_API_KEY")
    elif provider == "anthropic":
        together_key = _optional("TOGETHER_API_KEY")
        anthropic_key = _require("ANTHROPIC_API_KEY")
    elif provider == "openai":
        together_key = _optional("TOGETHER_API_KEY")
        anthropic_key = _optional("ANTHROPIC_API_KEY")
        _require("OPENAI_API_KEY")  # validate it exists; stored in env for openai SDK
    else:
        raise ConfigError(
            f"LLM_PROVIDER '{provider}' is not valid. "
            "Choose: together | anthropic | openai"
        )

    return Settings(
        llm_provider=provider,
        llm_model=_optional("LLM_MODEL", "moonshotai/Kimi-K2.5"),
        llm_temperature=float(_optional("LLM_TEMPERATURE", "0")),
        together_api_key=together_key,
        anthropic_api_key=anthropic_key,
        siril_bin=_optional("SIRIL_BIN", "siril-cli"),
        graxpert_bin=_optional("GRAXPERT_BIN", "graxpert"),
        starnet_bin=_require("STARNET_BIN"),
        starnet_weights=_require("STARNET_WEIGHTS"),
        pixel_size_um=float(v) if (v := _optional("PIXEL_SIZE_UM")) else None,
        langchain_tracing=_optional("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        langchain_api_key=_optional("LANGCHAIN_API_KEY"),
        langchain_project=_optional("LANGCHAIN_PROJECT", "astro-agent"),
        # Long-term memory (off by default — enable when agent quality is stable)
        memory_enabled=_optional("MEMORY_ENABLED", "false").lower() == "true",
        memory_db_path=_optional("MEMORY_DB_PATH", str(Path.home() / ".astro_agent" / "memory.db")),
        memory_embedding_model=_optional("MEMORY_EMBEDDING_MODEL", "qwen3-embedding"),
    )


# ── Dependency verification ────────────────────────────────────────────────────

def _check_siril(siril_bin: str) -> None:
    if not shutil.which(siril_bin):
        raise DependencyError(
            f"siril-cli not found at '{siril_bin}'. "
            "Install Siril ≥ 1.4 and ensure 'siril-cli' is on your PATH, "
            "or set SIRIL_BIN in .env to the full binary path. "
            "Download: https://siril.org/download/"
        )
    result = subprocess.run(
        [siril_bin, "--version"], capture_output=True, text=True, timeout=10
    )
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"(\d+)\.(\d+)", output)
    if not match:
        raise DependencyError(
            f"Could not parse Siril version from: {output!r}"
        )
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < (1, 4):
        raise DependencyError(
            f"Siril {major}.{minor} is too old. AstroAgent requires Siril ≥ 1.4. "
            "Upgrade: https://siril.org/download/"
        )


def configure_siril_for_equipment() -> str:
    """
    Write Siril demosaic preferences derived from equipment.toml.

    Siril has no per-command flag for Markesteijn pass count — it is a global
    preference in ~/.config/siril/siril.cfg. This function sets it correctly
    for the declared sensor type so the agent never needs to reason about it:

      xtrans → xtrans_passes=3  (Markesteijn 3-pass, best quality)
      bayer  → xtrans_passes=1  (Markesteijn not used for Bayer; reset to default)
      mono   → xtrans_passes=1  (no CFA demosaic needed)

    Returns the sensor_type string for logging.
    Raises ConfigError if equipment.toml is missing or sensor_type is unrecognised.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-reuse-def]

    equipment_path = Path(__file__).resolve().parent.parent / "equipment.toml"
    if not equipment_path.exists():
        raise ConfigError(
            f"equipment.toml not found at {equipment_path}. "
            "Create it from the project root template."
        )

    with open(equipment_path, "rb") as f:
        equipment = tomllib.load(f)

    sensor_type = equipment.get("camera", {}).get("sensor_type", "").lower()
    # sensor_type may be empty when equipment.toml is minimal (e.g. ZWO FITS
    # where sensor_type is detected from file headers, not equipment.toml).
    # Default to 1 xtrans pass (correct for Bayer and mono).
    if sensor_type and sensor_type not in ("xtrans", "bayer", "mono"):
        raise ConfigError(
            f"equipment.toml camera.sensor_type={sensor_type!r} is not recognised. "
            "Valid values: 'xtrans' | 'bayer' | 'mono'."
        )

    xtrans_passes = 3 if sensor_type == "xtrans" else 1

    cfg_dir = Path.home() / ".config" / "siril"
    cfg_file = cfg_dir / "siril.cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Parse existing config preserving all other settings, update xtrans_passes
    lines: list[str] = cfg_file.read_text().splitlines() if cfg_file.exists() else []
    in_core = False
    found_core = False
    found_key = False
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "[core]":
            in_core = True
            found_core = True
            out.append(line)
        elif stripped.startswith("[") and in_core:
            if not found_key:
                out.append(f"xtrans_passes={xtrans_passes}")
                found_key = True
            in_core = False
            out.append(line)
        elif in_core and stripped.startswith("xtrans_passes"):
            out.append(f"xtrans_passes={xtrans_passes}")
            found_key = True
        else:
            out.append(line)

    if not found_core:
        out.append("[core]")
    if not found_key:
        out.append(f"xtrans_passes={xtrans_passes}")

    cfg_file.write_text("\n".join(out) + "\n")
    return sensor_type


def _check_graxpert(graxpert_bin: str) -> None:
    if not shutil.which(graxpert_bin) and not Path(graxpert_bin).exists():
        raise DependencyError(
            f"GraXpert binary not found at '{graxpert_bin}'. "
            "Install GraXpert ≥ 3.0 and ensure the binary is on your PATH, "
            "or set GRAXPERT_BIN in .env. "
            "Download: https://github.com/Steffenhir/GraXpert/releases"
        )



def _check_starnet(starnet_bin: str, starnet_weights: str) -> None:
    """
    StarNet v2 on macOS is a standalone CLI binary downloaded from starnetastro.com.
    We call it directly via subprocess (not through Siril) using STARNET_BIN.
    The MPS-accelerated PyTorch build (starnet2) is preferred for Apple Silicon.

    See .env.example for full installation and code-signing instructions.
    """
    path = Path(starnet_bin)
    if not path.exists():
        raise DependencyError(
            f"StarNet binary not found at '{starnet_bin}'. "
            "Set STARNET_BIN in .env to the absolute path of your starnet2 executable.\n"
            "Installation:\n"
            "  1. Download StarNet v2 (MPS build) from: https://www.starnetastro.com/download/\n"
            "  2. chmod +x starnet2\n"
            "  3. xattr -d com.apple.quarantine starnet2\n"
            "  4. Allow in System Settings > Privacy & Security\n"
            "  5. codesign --force --sign - /path/to/starnet2\n"
            "  6. Set STARNET_BIN and STARNET_WEIGHTS in .env"
        )
    if not os.access(path, os.X_OK):
        raise DependencyError(
            f"StarNet binary at '{starnet_bin}' is not executable. "
            f"Run: chmod +x {starnet_bin}"
        )
    weights_path = Path(starnet_weights)
    if not weights_path.exists():
        raise DependencyError(
            f"StarNet weights file not found at '{starnet_weights}'. "
            "The weights file (StarNet2_weights.pt) ships in the same folder as the binary. "
            "Set STARNET_WEIGHTS in .env to its absolute path."
        )


def _check_exiftool() -> None:
    """ExifTool is installed system-wide (brew install exiftool) and must be on PATH."""
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        raise DependencyError(
            "ExifTool not found on PATH. "
            "Install with: brew install exiftool\n"
            "ExifTool is required for reading EXIF metadata from camera RAW files (T01)."
        )
    result = subprocess.run(
        [exiftool_path, "-ver"], capture_output=True, text=True, timeout=10
    )
    version_str = result.stdout.strip()
    try:
        major = int(version_str.split(".")[0])
        if major < 12:
            raise DependencyError(
                f"ExifTool {version_str} is too old. AstroAgent requires ExifTool ≥ 12. "
                "Upgrade: brew upgrade exiftool"
            )
    except (ValueError, IndexError):
        raise DependencyError(
            f"Could not parse ExifTool version from: {version_str!r}"
        )


def _check_python_libs() -> None:
    import packaging.version

    issues: list[str] = []

    try:
        import skimage
        if packaging.version.Version(skimage.__version__) < packaging.version.Version("0.22"):
            issues.append(f"scikit-image ≥ 0.22 required (installed: {skimage.__version__})")
    except ImportError:
        issues.append("scikit-image not installed  →  uv add scikit-image")

    try:
        import pywt
        if packaging.version.Version(pywt.__version__) < packaging.version.Version("1.6"):
            issues.append(f"PyWavelets ≥ 1.6 required (installed: {pywt.__version__})")
    except ImportError:
        issues.append("PyWavelets not installed  →  uv add PyWavelets")

    if issues:
        raise DependencyError(
            "Python library issues:\n  " + "\n  ".join(issues)
        )


def check_dependencies(settings: Settings | None = None) -> None:
    """
    Verify all external dependencies before the agent starts.
    Raises DependencyError with actionable install instructions on any failure.
    Called once from cli.py before invoking the graph.
    """
    if settings is None:
        settings = load_settings()

    _check_siril(settings.siril_bin)
    _check_graxpert(settings.graxpert_bin)
    _check_starnet(settings.starnet_bin, settings.starnet_weights)
    _check_exiftool()
    _check_python_libs()
    configure_siril_for_equipment()


# ── LLM factory (used by graph/planner.py in Phase 7) ─────────────────────────

def make_llm(settings: Settings | None = None):
    """
    Return a LangChain chat model bound to the configured provider.

    Together AI uses the OpenAI-compatible API (langchain-openai with
    base_url override). Switching to Anthropic or OpenAI is a one-line
    env change: set LLM_PROVIDER and the corresponding API key.
    """
    if settings is None:
        settings = load_settings()

    if settings.llm_provider == "together":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.together_api_key,       # type: ignore[arg-type]
            base_url="https://api.together.xyz/v1/",
            temperature=settings.llm_temperature,
            timeout=120,
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # optional dep
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,       # type: ignore[arg-type]
            temperature=settings.llm_temperature,
            timeout=120,
            model_kwargs={
                "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"},
            },
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=120,
        )

    raise ConfigError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


# ── Processing profiles ────────────────────────────────────────────────────────

PROFILE_DEFAULTS: dict[str, dict] = {
    "conservative": {
        "noise_reduction_modulation": 0.7,
        "stretch_method": "autostretch",
        "stretch_shadows_clip": -2.5,
        "saturation_amount": 0.2,
        "star_weight": 1.0,
        "deconvolution": False,
    },
    "balanced": {
        "noise_reduction_modulation": 0.85,
        "stretch_method": "ghs",
        "stretch_amount": 2.5,
        "saturation_amount": 0.4,
        "star_weight": 0.85,
        "deconvolution": True,
        "deconvolution_iterations": 10,
    },
    "aggressive": {
        "noise_reduction_modulation": 1.0,
        "stretch_method": "ghs",
        "stretch_amount": 4.0,
        "saturation_amount": 0.7,
        "star_weight": 0.6,
        "deconvolution": True,
        "deconvolution_iterations": 20,
    },
}
