"""
T24 — export_final

Export the finished image in distribution-ready formats with ICC color profiles.

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
    source_profile: str = Field(
        default="sRGBlinear",
        description=(
            "ICC profile to assign to the input FITS before converting. "
            "Can be a built-in name or a path to an external ICC file."
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
    source_profile: str = "sRGBlinear",
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
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    # Auto-detect source profile from image linearity when the caller used
    # the default.  After stretching the data is non-linear — assigning
    # sRGBlinear would re-apply gamma on top of the stretch, washing out
    # the image.  sRGB tells Siril the data is already gamma-encoded.
    is_linear = state.get("metrics", {}).get("is_linear_estimate", True)
    if source_profile == "sRGBlinear" and not is_linear:
        source_profile = "sRGB"

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
    exported_files = []

    for fmt in formats:
        output_stem = f"{stem}{fmt.filename_suffix}"

        result_path = _export_one(
            stem=stem,
            working_dir=working_dir,
            fmt=fmt,
            output_stem=output_stem,
            source_profile=source_profile,
        )

        final_path = export_dir / result_path.name
        result_path.rename(final_path)

        file_size_mb = round(final_path.stat().st_size / (1024 * 1024), 2)
        exported_files.append({
            "path": str(final_path),
            "format": fmt.type,
            "icc_profile": fmt.icc_profile,
            "file_size_mb": file_size_mb,
        })

    return Command(update={
        "metadata": {
            **state["metadata"],
            "export_done": True,
            "exported_files": exported_files,
        },
        "messages": [ToolMessage(
            content=json.dumps({"exported_files": exported_files}, indent=2),
            tool_call_id=tool_call_id,
        )],
    })
