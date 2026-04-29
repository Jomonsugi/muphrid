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

# Agent-VLM preview defaults. The human-facing preview stays at the larger
# size for the Gradio panel; the agent gets a smaller sibling so the per-cycle
# vision token cost stays reasonable when the working image is auto-injected
# every model.invoke. Roughly 1024px long-side at JPG q=85 ≈ 450–550 vision
# tokens on Anthropic Claude — enough resolution for the structure-scale
# judgments astrophotography decisions need (gradient direction, vignetting,
# nebula shape, color cast, "do these stars look right") without paying for
# pixel-level detail the analytical metrics cover precisely.
_AGENT_VLM_WIDTH = 1024
_AGENT_VLM_QUALITY = 85


def generate_preview(
    working_dir: str,
    fits_path: str,
    width: int = 1920,
    format: str = "jpg",
    auto_stretch_linear: bool = True,
    quality: int = 95,
    annotation: str | None = None,
    write_agent_sidecar: bool = True,
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
    write_agent_sidecar : bool
        When True (default) and format='jpg', also write a smaller
        `<stem>_vlm.jpg` sidecar at _AGENT_VLM_WIDTH / _AGENT_VLM_QUALITY for
        the agent's vision input. The sidecar reuses the same Siril output so
        only the resize/save is repeated. Skipped for PNG output (the agent
        path always uses JPG) and skipped when the request is itself already
        smaller than the agent target (no point producing a sidecar that
        equals the human preview).

    Returns
    -------
    dict
        preview_path           : absolute path to the human-facing preview
        is_auto_stretched      : whether autostretch was applied
        agent_preview_path     : absolute path to the agent VLM preview, if
                                 a sidecar was written; otherwise None
    """
    img_path = Path(fits_path)
    if not img_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    preview_dir = Path(working_dir) / "previews"
    preview_dir.mkdir(exist_ok=True)

    stem = img_path.stem
    render_mode = "linear_autostretch" if auto_stretch_linear else "display_faithful"
    preview_stem = f"preview_{stem}_{render_mode}"

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

    # Agent VLM sidecar — produced from the human preview (already on disk),
    # never re-rendered from FITS. Same colour pipeline, smaller resolution.
    agent_preview_path: str | None = None
    if (
        write_agent_sidecar
        and format == "jpg"
        and width > _AGENT_VLM_WIDTH
    ):
        try:
            agent_preview = preview_dir / f"{preview_stem}_vlm.jpg"
            _resize_to_jpg(
                src=final_path,
                dst=agent_preview,
                width=_AGENT_VLM_WIDTH,
                quality=_AGENT_VLM_QUALITY,
            )
            agent_preview_path = str(agent_preview)
        except Exception as e:
            # The human preview is the load-bearing artifact; the sidecar is
            # an optimization. Don't fail the whole call if it can't be
            # produced (e.g. PIL import failure on a slim install).
            import logging
            logging.getLogger(__name__).warning(
                f"generate_preview: VLM sidecar write failed (non-fatal): {e}"
            )

    # Clean up temp file if it differs from final path
    if temp_file != final_path and temp_file.exists():
        temp_file.unlink()

    return {
        "preview_path": str(final_path),
        "is_auto_stretched": auto_stretch_linear,
        "agent_preview_path": agent_preview_path,
    }


def _resize_to_jpg(
    src: Path,
    dst: Path,
    width: int,
    quality: int,
) -> None:
    """
    Resize an existing JPG/PNG to a smaller JPG. Used to produce the agent
    VLM sidecar from the human-facing preview without re-running Siril.
    """
    from PIL import Image

    img = Image.open(src)
    if img.width > width:
        ratio = width / img.width
        new_size = (width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
    img.save(dst, format="JPEG", quality=quality, optimize=True)


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
