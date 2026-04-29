"""
T24 — export_final + commit_export

Export the finished image in distribution-ready formats with ICC color profiles.

Source-profile selection is driven by `state.metadata.image_space` — the
authoritative render-state contract. When image_space="display" the source
profile is sRGB (data is already gamma-encoded after stretch); when
image_space="linear" it is sRGBlinear (Siril applies gamma on convert).
The `metrics.is_linear_estimate` heuristic is NOT consulted — the human
might have approved a stretched variant whose heuristic happened to read
as linear, and binding the export to the heuristic would write a different
artifact than what was approved. State authority means state, full stop;
the heuristic stays diagnostic.

Tentative-export pattern (when the T24_export HITL gate is enabled):

  1. export_final runs with `tentative=True` (the agent flips this when
     the gate is on; otherwise it defaults to False / direct export).
     Files are written to `<output_dir>/.tentative_<stem>/`.
  2. The tool emits a ReviewSession proposal whose visual_path points
     at the actual rendered JPG export (the artifact the human will
     ultimately see — not a FITS-derived auto-preview).
  3. The HITL gate fires; the human approves the proposal.
  4. The agent calls commit_export, which moves the tentative files
     into the canonical export_dir and clears state.metadata.tentative_export.

When the gate is disabled (autonomous), export_final writes directly
to the export_dir with no tentative staging.

Siril commands (verified against Siril 1.4 CLI docs):
    icc_assign profile               — Built-ins: sRGB, sRGBlinear, Rec2020,
                                        Rec2020linear, working, linear,
                                        graysrgb, grayrec2020, graylinear
                                        OR path to external ICC file
    icc_convert_to profile [intent]  — Same profiles + intent: perceptual,
                                        relative, saturation, absolute
    savetif <stem> [-astro] [-deflate]   — 16-bit TIFF
    savetif8 <stem> [-astro] [-deflate]  — 8-bit TIFF
    savetif32 <stem> [-astro] [-deflate] — 32-bit TIFF
    savejpg <stem> [quality]             — JPEG (100=best)
    savepng <stem>                       — PNG (16-bit if source ≥16-bit)
    savejxl <stem> [-effort=] [-quality=] [-8bit]  — JPEG XL
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script


# ── Constants ──────────────────────────────────────────────────────────────────

BUILTIN_PROFILES = {
    "sRGB", "sRGBlinear", "Rec2020", "Rec2020linear",
    "working", "linear",
    "graysrgb", "grayrec2020", "graylinear",
}

VALID_FORMATS = {"tiff8", "tiff16", "tiff32", "jpg", "png", "jxl"}
VALID_INTENTS = {"perceptual", "relative", "saturation", "absolute"}

DEFAULT_FORMATS = [
    {
        "type": "tiff16",
        "icc_profile": "Rec2020",
        "filename_suffix": "_master",
        "quality": 95,
        "deflate": True,
    },
    {
        "type": "jpg",
        "icc_profile": "sRGB",
        "filename_suffix": "_web",
        "quality": 95,
        "deflate": False,
    },
]

# Keep for backward compat with verify scripts
VALID_PROFILES = BUILTIN_PROFILES


# ── Pydantic input schema ──────────────────────────────────────────────────────

class FormatSpec(BaseModel):
    type: str = Field(
        description=(
            "Output format: "
            "'tiff8' (8-bit TIFF), "
            "'tiff16' (16-bit TIFF, archival), "
            "'tiff32' (32-bit TIFF, maximum precision), "
            "'jpg' (lossy, web sharing), "
            "'png' (lossless, 16-bit if source ≥16-bit), "
            "'jxl' (JPEG XL — near-lossless with small file size)."
        )
    )
    icc_profile: str = Field(
        default="sRGB",
        description=(
            "ICC color profile for the export. "
            "Built-in profiles: sRGB, sRGBlinear, Rec2020, Rec2020linear, "
            "working, linear, graysrgb, grayrec2020, graylinear. "
            "Can also be an absolute path to an external .icc/.icm file "
            "(e.g. AdobeRGB1998.icc)."
        ),
    )
    filename_suffix: str = Field(
        default="",
        description="Appended to the source image stem for the export filename.",
    )
    quality: int = Field(
        default=95,
        description="JPG quality (1–100). Ignored for TIFF and PNG.",
    )
    deflate: bool = Field(
        default=True,
        description="Apply deflate compression to TIFF. No quality loss, smaller files.",
    )
    astro_tiff: bool = Field(
        default=False,
        description=(
            "Save as Astro-TIFF format (-astro). Preserves FITS keywords in "
            "TIFF metadata. Recommended for archival TIFF exports."
        ),
    )
    rendering_intent: str = Field(
        default="perceptual",
        description="ICC rendering intent: perceptual, relative, saturation, absolute.",
    )
    jxl_quality: float = Field(
        default=9.0,
        description=(
            "JPEG XL quality (0.0–10.0). 10.0 = mathematically lossless, "
            "9.0 = visually lossless (default), 7.0+ for high quality."
        ),
    )
    jxl_effort: int = Field(
        default=7,
        description=(
            "JPEG XL compression effort (1–9). Higher = smaller files but slower. "
            "7 is a good default. Values > 7 have diminishing returns."
        ),
    )
    jxl_8bit: bool = Field(
        default=False,
        description="Force 8-bit output for JPEG XL (-8bit).",
    )


class ExportFinalInput(BaseModel):
    formats: list[FormatSpec] = Field(
        default_factory=lambda: [FormatSpec(**f) for f in DEFAULT_FORMATS],
        description=(
            "List of export format specifications. "
            "Default: Rec2020 TIFF16 + sRGB JPG."
        ),
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory for exported files. Defaults to working_dir/export/.",
    )
    source_profile: str | None = Field(
        default=None,
        description=(
            "ICC profile to assign to the input FITS before converting. "
            "If None (default), the source profile is derived from "
            "state.metadata.image_space — 'sRGB' for display-space data "
            "(post-stretch) or 'sRGBlinear' for linear data. State is the "
            "authoritative contract; explicit override is for unusual "
            "workflows (e.g. importing pre-color-managed TIFs). "
            "Built-in names: sRGB, sRGBlinear, Rec2020, Rec2020linear, "
            "working, linear, graysrgb, grayrec2020, graylinear. May also "
            "be an absolute path to an external ICC file."
        ),
    )
    tentative: bool = Field(
        default=False,
        description=(
            "When True, write export artifacts to "
            "<output_dir>/.tentative_<stem>/ and record their paths in "
            "state.metadata.tentative_export so the HITL gate can present "
            "the actual rendered JPG before committing. The agent should "
            "set this when the T24_export HITL gate is enabled. After "
            "human approval, call commit_export to move the tentative "
            "files into output_dir. When False (autonomous), files go "
            "directly into output_dir."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _profile_arg(profile: str) -> str:
    """Return the profile argument, quoting paths with spaces."""
    if profile in BUILTIN_PROFILES:
        return profile
    if " " in profile:
        return f'"{profile}"'
    return profile


def _export_one(
    stem: str,
    working_dir: str,
    fmt: FormatSpec,
    output_stem: str,
    source_profile: str = "sRGBlinear",
) -> Path:
    """Run a single Siril export script for one format/profile combination."""
    profile = _profile_arg(fmt.icc_profile)
    intent = fmt.rendering_intent if fmt.rendering_intent in VALID_INTENTS else "perceptual"
    src_prof = _profile_arg(source_profile)

    commands = [
        f"load {stem}",
        f"icc_assign {src_prof}",
        f"icc_convert_to {profile} {intent}",
    ]

    fmt_type = fmt.type.lower()
    tiff_flags = ""
    if fmt.astro_tiff:
        tiff_flags += " -astro"
    if fmt.deflate:
        tiff_flags += " -deflate"

    if fmt_type == "tiff8":
        commands.append(f"savetif8 {output_stem}{tiff_flags}")
        ext = ".tif"
    elif fmt_type == "tiff16":
        commands.append(f"savetif {output_stem}{tiff_flags}")
        ext = ".tif"
    elif fmt_type == "tiff32":
        commands.append(f"savetif32 {output_stem}{tiff_flags}")
        ext = ".tif"
    elif fmt_type == "jpg":
        quality = max(1, min(100, fmt.quality))
        commands.append(f"savejpg {output_stem} {quality}")
        ext = ".jpg"
    elif fmt_type == "png":
        commands.append(f"savepng {output_stem}")
        ext = ".png"
    elif fmt_type == "jxl":
        jxl_cmd = f"savejxl {output_stem}"
        jxl_cmd += f" -quality={fmt.jxl_quality}"
        jxl_cmd += f" -effort={fmt.jxl_effort}"
        if fmt.jxl_8bit:
            jxl_cmd += " -8bit"
        commands.append(jxl_cmd)
        ext = ".jxl"
    else:
        raise ValueError(f"Unknown export format type: {fmt_type!r}")

    run_siril_script(commands, working_dir=working_dir, timeout=120)

    output_path = Path(working_dir) / f"{output_stem}{ext}"
    if not output_path.exists():
        raise FileNotFoundError(
            f"export_final: Siril did not produce: {output_path}"
        )
    return output_path


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=ExportFinalInput)
def export_final(
    formats: list[FormatSpec] | None = None,
    output_dir: str | None = None,
    source_profile: str | None = None,
    tentative: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Export the finished image in distribution-ready formats with ICC profiles.

    Always call with at least two formats:
    - tiff16 + Rec2020: archival master (wide gamut, lossless)
    - jpg + sRGB: web sharing (correct color on consumer screens)

    Additional format options:
    - tiff8: 8-bit TIFF for lightweight archives or when software requires 8-bit
    - tiff32: 32-bit TIFF for maximum precision (large files)
    - jxl: JPEG XL — near-lossless at much smaller sizes than TIFF
    - png: lossless 16-bit for web contexts requiring transparency

    Astro-TIFF (-astro): preserves FITS metadata in TIFF headers. Recommended
    for archival exports that may be re-imported into astro software.

    ICC profiles can be Siril built-ins (sRGB, Rec2020, graysrgb, etc.) or
    absolute paths to external ICC/ICM files for AdobeRGB or custom profiles.

    For mono images, use gray profiles: graysrgb, grayrec2020, graylinear.

    Source-profile selection: by default, derived from state.metadata.image_space
    (the authoritative render-state contract). 'display' → sRGB; 'linear' →
    sRGBlinear. State is refused if missing or invalid — the heuristic
    `metrics.is_linear_estimate` is no longer consulted because it can
    disagree with the artifact the human approved. Override `source_profile`
    explicitly only for unusual workflows.

    Tentative mode: when `tentative=True`, files are written to
    <output_dir>/.tentative_<stem>/ and recorded in
    state.metadata.tentative_export. Use with the T24_export HITL gate so
    the human reviews the rendered JPG, then call commit_export to move
    the tentative files into output_dir.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    # State authority: source_profile is selected from image_space, not the
    # diagnostic heuristic. Refuse if state is missing/invalid — exporting
    # against an unknown source profile would silently produce a
    # mis-rendered artifact (sRGBlinear on a stretched image re-applies
    # gamma and washes the image out; sRGB on linear data clips the
    # shadows). See Metadata.image_space and CLAUDE.md.
    incoming_image_space = state.get("metadata", {}).get("image_space")
    if incoming_image_space not in ("linear", "display"):
        raise RuntimeError(
            "export_final: state.metadata.image_space is missing or invalid "
            f"(got {incoming_image_space!r}). The export needs the "
            "authoritative render-state to choose the correct ICC source "
            "profile. Every writer of paths.current_image must also write "
            "metadata.image_space (enforced by the registry drift check). "
            "This looks like a legacy checkpoint or a writer that skipped "
            "its bookkeeping. Refusing to guess — restart from a fresh "
            "checkpoint."
        )

    if source_profile is None:
        # display → sRGB (data already gamma-encoded by the stretch);
        # linear → sRGBlinear (Siril applies gamma on convert).
        source_profile = "sRGBlinear" if incoming_image_space == "linear" else "sRGB"

    if formats is None:
        formats = [FormatSpec(**f) for f in DEFAULT_FORMATS]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_dir:
        export_dir = Path(output_dir)
    else:
        export_dir = Path(working_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem

    # Tentative writes land in a per-stem staging dir under export_dir.
    # commit_export moves them into export_dir on approval. Disambiguating
    # by stem keeps two parallel review cycles from clobbering each other
    # (rare, but possible when the agent re-exports after revision).
    if tentative:
        write_dir = export_dir / f".tentative_{stem}"
        write_dir.mkdir(parents=True, exist_ok=True)
    else:
        write_dir = export_dir
    exported_files = []
    jpg_preview_path: str | None = None

    for fmt in formats:
        output_stem = f"{stem}{fmt.filename_suffix}"

        result_path = _export_one(
            stem=stem,
            working_dir=working_dir,
            fmt=fmt,
            output_stem=output_stem,
            source_profile=source_profile,
        )

        final_path = write_dir / result_path.name
        result_path.rename(final_path)

        file_size_mb = round(final_path.stat().st_size / (1024 * 1024), 2)
        exported_files.append({
            "path": str(final_path),
            "format": fmt.type,
            "icc_profile": fmt.icc_profile,
            "file_size_mb": file_size_mb,
        })
        # Keep the first JPG path for the HITL preview — the human reviews
        # the actual rendered JPG, not a FITS-derived auto-stretch. If no
        # JPG was requested, jpg_preview_path stays None and the gate uses
        # the FITS preview as a fallback (still better than a mock path).
        if jpg_preview_path is None and fmt.type.lower() == "jpg":
            jpg_preview_path = str(final_path)

    summary: dict = {
        "exported_files": exported_files,
        "image_space": incoming_image_space,
        "source_profile": source_profile,
        "tentative": tentative,
        "preview_jpg": jpg_preview_path,
    }
    # Surface the JPG (when present) as both output_path and preview_path
    # so the HITL gate's variant-snapshot logic can pick up THE actual
    # rendered artifact and present it to the human, not a FITS-derived
    # auto-stretched preview. nodes._extract_variant_paths probes
    # _VARIANT_FILE_KEYS in priority order; output_path is first.
    if jpg_preview_path is not None:
        summary["output_path"] = jpg_preview_path
        summary["preview_path"] = jpg_preview_path

    metadata_delta: dict = {
        "image_space": incoming_image_space,
        "export_done": not tentative,
    }
    if tentative:
        metadata_delta["tentative_export"] = {
            "stem": stem,
            "write_dir": str(write_dir),
            "final_dir": str(export_dir),
            "exported_files": exported_files,
            "preview_jpg": jpg_preview_path,
        }
    else:
        # Direct (non-tentative) export — populate exported_files and clear
        # any lingering tentative_export from a prior aborted gate.
        metadata_delta["exported_files"] = exported_files
        metadata_delta["tentative_export"] = None

    return Command(update={
        "metadata": metadata_delta,
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            tool_call_id=tool_call_id,
        )],
    })


# ── commit_export tool ────────────────────────────────────────────────────────

class CommitExportInput(BaseModel):
    note: str | None = Field(
        default=None,
        description=(
            "Optional human-supplied note recorded with the commit, e.g. "
            "the rationale for accepting this export. Has no functional "
            "effect; preserved in the tool message for audit."
        ),
    )


@tool(args_schema=CommitExportInput)
def commit_export(
    note: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Promote a tentative export to its final destination.

    Reads `state.metadata.tentative_export` (set by `export_final(tentative=True)`),
    moves each tentative file from the staging directory into the canonical
    export_dir, removes the empty staging directory, and clears the
    tentative_export marker. Sets `metadata.export_done=True` and
    `metadata.exported_files` to the final paths.

    Refuses if no tentative_export is present — the caller must run
    export_final(tentative=True) first. Refuses if any tentative file is
    missing on disk (would-be silent data loss otherwise). Idempotent
    against partial moves: if a tentative file's destination already
    exists with the same content, the move is treated as already-complete.
    """
    metadata = state.get("metadata", {}) or {}
    tentative = metadata.get("tentative_export")
    if not isinstance(tentative, dict):
        raise RuntimeError(
            "commit_export: state.metadata.tentative_export is not set. "
            "Run export_final(tentative=True) first to produce a staged "
            "export, then call commit_export after the human approves."
        )

    write_dir = Path(tentative["write_dir"])
    final_dir = Path(tentative["final_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)

    moved: list[dict] = []
    for entry in tentative.get("exported_files", []):
        src = Path(entry["path"])
        if not src.exists():
            raise FileNotFoundError(
                f"commit_export: tentative file missing at commit time: {src}. "
                "Refusing to silently complete a partial export — investigate "
                "before retrying."
            )
        dst = final_dir / src.name
        if dst.exists() and dst.resolve() == src.resolve():
            # Already in place (e.g. write_dir == final_dir for some reason).
            moved_entry = dict(entry)
            moved.append(moved_entry)
            continue
        if dst.exists():
            # Conservative: don't clobber an unrelated final-dir file.
            raise FileExistsError(
                f"commit_export: destination already exists and is not the "
                f"tentative source: {dst}. Resolve manually before retrying."
            )
        src.rename(dst)
        moved_entry = dict(entry)
        moved_entry["path"] = str(dst)
        moved.append(moved_entry)

    # Best-effort cleanup of the empty staging directory.
    try:
        if write_dir.exists() and not any(write_dir.iterdir()):
            write_dir.rmdir()
    except OSError:
        # Non-fatal: leftover staging dir is mildly untidy, not a data risk.
        pass

    summary = {
        "committed_files": moved,
        "final_dir": str(final_dir),
        "note": note,
    }

    return Command(update={
        "metadata": {
            "export_done": True,
            "exported_files": moved,
            "tentative_export": None,
        },
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            tool_call_id=tool_call_id,
        )],
    })
