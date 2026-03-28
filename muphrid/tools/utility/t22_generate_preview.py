"""
T22 — generate_preview  [INTERNAL FUNCTION — NOT agent-callable]

Creates a JPG or PNG preview from a FITS file for HITL display. For linear
images, applies a temporary autostretch for visualization without modifying
the FITS source.

This is NOT a @tool-decorated function. It must not appear in any PHASE_TOOLS
list. It is called exclusively by:
  1. auto_hitl_check() in the tool executor post-hook (for tools with
     requires_visual_review=True: remove_gradient, deconvolution, star_removal)
  2. Mandatory HITL nodes (stretch_hitl, final_hitl)

The planner's context is always text-only metrics — preview generation never
happens in the planner path.

Siril commands (verified against Siril 1.4 CLI docs):
    load <stem>
    autostretch [-linked] [shadowsclip [targetbg]]
    savejpg <filename> [quality]
    savepng <filename>
"""

from __future__ import annotations

import re
from pathlib import Path

from muphrid.tools._siril import run_siril_script


# ── Core function ──────────────────────────────────────────────────────────────

def generate_preview(
    working_dir: str,
    fits_path: str,
    width: int = 1920,
    format: str = "jpg",
    auto_stretch_linear: bool = True,
    quality: int = 95,
    annotation: str | None = None,
) -> dict:
    """
    Generate a preview image from a FITS file.

    Parameters
    ----------
    working_dir : str
        Absolute path to the Siril working directory.
    fits_path : str
        Absolute path to the FITS file to preview.
    width : int
        Target preview width in pixels. Image is resized proportionally.
    format : str
        Output format: 'jpg' or 'png'.
    auto_stretch_linear : bool
        Apply autostretch before saving. Always True for linear images —
        without it, the preview appears nearly black.
    quality : int
        JPG compression quality (1–100). Ignored for PNG.
    annotation : str | None
        Optional text label drawn on the preview (e.g. "Variant A: Gentle").

    Returns
    -------
    dict
        preview_path : str — absolute path to the generated preview file
        is_auto_stretched : bool — whether autostretch was applied
    """
    img_path = Path(fits_path)
    if not img_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    preview_dir = Path(working_dir) / "previews"
    preview_dir.mkdir(exist_ok=True)

    stem = img_path.stem
    preview_stem = f"preview_{stem}"

    commands = [f"load {stem}"]

    if auto_stretch_linear:
        # -linked preserves white balance after color calibration
        commands.append("autostretch -linked")

    # Save to working_dir root first (Siril's default working directory)
    temp_stem = preview_stem
    if format == "png":
        commands.append(f"savepng {temp_stem}")
        temp_file = Path(working_dir) / f"{temp_stem}.png"
    else:
        commands.append(f"savejpg {temp_stem} {quality}")
        temp_file = Path(working_dir) / f"{temp_stem}.jpg"

    run_siril_script(commands, working_dir=working_dir, timeout=60)

    if not temp_file.exists():
        raise FileNotFoundError(f"Siril did not produce preview: {temp_file}")

    # Resize and optionally annotate using Pillow
    final_path = preview_dir / temp_file.name
    _resize_and_annotate(temp_file, final_path, width=width, annotation=annotation)

    # Clean up temp file if it differs from final path
    if temp_file != final_path and temp_file.exists():
        temp_file.unlink()

    return {
        "preview_path": str(final_path),
        "is_auto_stretched": auto_stretch_linear,
    }


def _resize_and_annotate(
    src: Path,
    dst: Path,
    width: int,
    annotation: str | None,
) -> None:
    """Resize image to target width and optionally draw annotation text."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(src)

    if img.width > width:
        ratio = width / img.width
        new_size = (width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    if annotation:
        draw = ImageDraw.Draw(img)
        # Use default font — keeps dependencies minimal
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=28)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Shadow for readability on dark backgrounds
        draw.text((12, 12), annotation, fill=(0, 0, 0), font=font)
        draw.text((10, 10), annotation, fill=(255, 255, 255), font=font)

    ext = dst.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        # Convert RGBA → RGB if needed (Siril sometimes adds alpha)
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        img.save(dst, format="JPEG", quality=95)
    else:
        img.save(dst)
