"""
T24 — export_final

Export the finished image in distribution-ready formats with ICC color
profiles. Always produce at least two exports:
  - 16-bit TIFF with Rec2020 profile (archival master, wide gamut)
  - JPG with sRGB profile (web sharing, correct display on consumer screens)

Siril commands (verified against Siril 1.4 CLI docs):
    load <stem>
    icc_convert_to <profile> [intent]   — Built-ins: sRGB, sRGBlinear, Rec2020, Rec2020linear
    savetif <stem> [-astro] [-deflate]  — 16-bit TIFF
    savetif32 <stem> [-astro] [-deflate]— 32-bit TIFF
    savejpg <stem> [quality]            — JPEG (100=best, lossy)
    savepng <stem>                      — PNG (16-bit if image is ≥16-bit)

ICC profile notes:
  - 'sRGB': standard sRGB gamma — correct for web/social sharing
  - 'Rec2020': wide-gamut linear approximation — archival/print master
  - 'sRGBlinear': linear light sRGB — for HDR pipelines
  - 'AdobeRGB' is NOT a Siril built-in — use Rec2020 as the wide-gamut choice

Each format runs in a separate Siril script call so profile conversion
does not bleed between formats.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Constants ──────────────────────────────────────────────────────────────────

VALID_PROFILES = {"sRGB", "sRGBlinear", "Rec2020", "Rec2020linear"}
VALID_FORMATS  = {"tiff16", "tiff32", "jpg", "png"}
VALID_INTENTS  = {"perceptual", "relative", "saturation", "absolute"}

# Default: archival TIFF (Rec2020 for wide gamut) + web JPG (sRGB)
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


# ── Pydantic input schema ──────────────────────────────────────────────────────

class FormatSpec(BaseModel):
    type: str = Field(
        description=(
            "Output format: "
            "'tiff16' (16-bit TIFF, archival quality), "
            "'tiff32' (32-bit TIFF, maximum precision), "
            "'jpg' (lossy, web sharing), "
            "'png' (lossless, 16-bit if source is ≥16-bit)."
        )
    )
    icc_profile: str = Field(
        default="sRGB",
        description=(
            "ICC color profile for the export. "
            "Valid built-in profiles: sRGB (web/consumer), sRGBlinear, "
            "Rec2020 (wide-gamut archival, recommended for TIFF master), "
            "Rec2020linear. "
            "Note: AdobeRGB is not a Siril built-in — use Rec2020 instead."
        ),
    )
    filename_suffix: str = Field(
        default="",
        description=(
            "Appended to the source image stem for the export filename. "
            "Recommended: '_master' for TIFF, '_web' for JPG."
        ),
    )
    quality: int = Field(
        default=95,
        description="JPG quality (1–100). Ignored for TIFF and PNG.",
    )
    deflate: bool = Field(
        default=True,
        description="Apply deflate compression to TIFF. No quality loss, smaller files.",
    )
    rendering_intent: str = Field(
        default="perceptual",
        description=(
            "ICC color transform rendering intent: "
            "perceptual (default, best for photographs), "
            "relative (relative colorimetric), "
            "saturation, absolute."
        ),
    )


class ExportFinalInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the final FITS image to export. "
            "Must be non-linear (stretched). This is the last step — "
            "call after all processing is complete."
        )
    )
    formats: list[FormatSpec] = Field(
        default_factory=lambda: [FormatSpec(**f) for f in DEFAULT_FORMATS],
        description=(
            "List of export format specifications. "
            "Default: Rec2020 TIFF16 (archival master) + sRGB JPG (web). "
            "Add formats as needed but always include both defaults."
        ),
    )
    output_dir: str | None = Field(
        default=None,
        description=(
            "Directory for exported files. Defaults to working_dir/export/. "
            "Created if it does not exist."
        ),
    )
    source_profile: str = Field(
        default="sRGBlinear",
        description=(
            "ICC profile to assign to the input FITS before converting. "
            "Pipeline FITS typically have no embedded profile; this sets the "
            "working color space. 'sRGBlinear' is correct for linear or "
            "stretched data processed in sRGB primaries."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _export_one(
    stem: str,
    working_dir: str,
    fmt: FormatSpec,
    output_stem: str,
    source_profile: str = "sRGBlinear",
) -> Path:
    """Run a single Siril export script for one format/profile combination.

    Assigns source_profile first to ensure the image has a working color space
    before converting.  Without this, icc_convert_to fails on images that have
    no ICC profile embedded (common for pipeline FITS).
    """
    profile = fmt.icc_profile if fmt.icc_profile in VALID_PROFILES else "sRGB"
    intent = fmt.rendering_intent if fmt.rendering_intent in VALID_INTENTS else "perceptual"

    commands = [
        f"load {stem}",
        f"icc_assign {source_profile}",
        f"icc_convert_to {profile} {intent}",
    ]

    fmt_type = fmt.type.lower()
    if fmt_type == "tiff16":
        deflate_flag = " -deflate" if fmt.deflate else ""
        commands.append(f"savetif {output_stem}{deflate_flag}")
        ext = ".tif"
    elif fmt_type == "tiff32":
        deflate_flag = " -deflate" if fmt.deflate else ""
        commands.append(f"savetif32 {output_stem}{deflate_flag}")
        ext = ".tif"
    elif fmt_type == "jpg":
        quality = max(1, min(100, fmt.quality))
        commands.append(f"savejpg {output_stem} {quality}")
        ext = ".jpg"
    elif fmt_type == "png":
        commands.append(f"savepng {output_stem}")
        ext = ".png"
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
    working_dir: str,
    image_path: str,
    formats: list[FormatSpec] | None = None,
    output_dir: str | None = None,
    source_profile: str = "sRGBlinear",
) -> dict:
    """
    Export the finished image in distribution-ready formats with ICC profiles.

    Always call with at least two formats:
    - tiff16 + Rec2020: archival master (wide gamut, lossless)
    - jpg + sRGB: web sharing (correct color on consumer screens)

    Each format is exported in a separate Siril run so profile conversions
    are independent and cannot interfere with each other.

    Note on profiles:
    - Use Rec2020 (not AdobeRGB) for wide-gamut archival — it is a Siril
      built-in. AdobeRGB requires an external ICC file path.
    - Apply icc_convert_to before saving — without it, Siril saves with the
      working color profile, which may be linear (incorrect for JPG).
    """
    if formats is None:
        formats = [FormatSpec(**f) for f in DEFAULT_FORMATS]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Set up export directory
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

        # Move to export directory
        final_path = export_dir / result_path.name
        result_path.rename(final_path)

        file_size_mb = round(final_path.stat().st_size / (1024 * 1024), 2)
        exported_files.append({
            "path": str(final_path),
            "format": fmt.type,
            "icc_profile": fmt.icc_profile,
            "file_size_mb": file_size_mb,
        })

    return {"exported_files": exported_files}
