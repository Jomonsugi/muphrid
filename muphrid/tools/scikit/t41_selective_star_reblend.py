"""
T41 — selective_star_reblend

Recombine `paths.starless_image` and `paths.star_mask` from t15 with a
per-source weight map W in [0,1] painted on the star mask. The kept set
of sources receives W=1.0; non-kept sources receive W=`suppress_strength`.

result = starless + dilated_star_mask * W

Detection runs on the star_mask itself (StarNet's mask isolates star
contribution, so peak detection on it is more robust than re-running
detection on the starred original — and the starred original isn't
always retained on disk).

Optional `confine_to_region_mask` reads `paths.latest_mask` (last output
of `create_mask`) and applies the per-source weight rules only inside
the region; outside the region, W=1.0 (full restoration).

Backend: pure numpy + Astropy + skimage. No Siril.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from astropy.io import fits as astropy_fits
from astropy.stats import mad_std
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, disk

from muphrid.graph.state import AstroState


# ── Pydantic input schema ──────────────────────────────────────────────────────


class SelectiveStarReblendInput(BaseModel):
    mode: Literal["brightness_priority", "color_priority", "balanced"] = Field(
        default="brightness_priority",
        description=(
            "Per-source ranking rule. "
            "'brightness_priority': score = peak_norm * (1 + 0.3 * chroma_norm). "
            "'color_priority':      score = chroma_norm * (1 + 0.3 * peak_norm). "
            "'balanced':            score = 0.5 * (peak_rank + chroma_rank), "
            "where _rank is the source's percentile rank in [0,1]."
        ),
    )
    keep_fraction: float = Field(
        default=0.1,
        description=(
            "Fraction of detected sources receiving W=1.0 (full restoration), "
            "ranked by score. 0.05 keeps the top 5% at full weight; remaining "
            "sources receive `suppress_strength`. Range [0, 1]."
        ),
    )
    suppress_strength: float = Field(
        default=0.3,
        description=(
            "Weight applied to non-kept sources. 0.0 removes them entirely; "
            "1.0 makes the suppression a no-op. Range [0, 1]."
        ),
    )
    core_radius_factor: float = Field(
        default=1.5,
        description=(
            "Per-source disk radius in units of fwhm. 1.5 covers PSF + first "
            "ring; 2.0 covers wider halos; lower values produce sharper "
            "weight transitions."
        ),
    )
    feather_sigma_px: float = Field(
        default=2.0,
        description=(
            "Gaussian sigma applied to the weight map W after disk painting "
            "and before blending. Larger sigma blurs transitions across "
            "neighboring sources; 0 disables feathering."
        ),
    )
    mask_dilation_px: int = Field(
        default=0,
        description=(
            "Pixels of binary dilation applied to the StarNet mask before "
            "blending. >0 grows the support of the star contribution; "
            "useful when StarNet edges leave thin halos in the starless. "
            "0 disables."
        ),
    )
    confine_to_region_mask: bool = Field(
        default=False,
        description=(
            "When true, the per-source weight rules apply only where "
            "paths.latest_mask > 0.5; outside the region, W=1.0 (full "
            "restoration). Requires create_mask to have been called first."
        ),
    )
    threshold_sigma: float = Field(
        default=5.0,
        description=(
            "Detection threshold in MAD-noise units above background, "
            "applied to the star_mask luminance. 5.0: standard."
        ),
    )
    fwhm_guess: float = Field(
        default=3.0,
        description="Initial FWHM guess in pixels for IRAFStarFinder.",
    )
    min_separation_fwhm: float = Field(
        default=2.0,
        description="Minimum allowed source separation in fwhm units.",
    )
    max_sources: int = Field(
        default=5000,
        description=(
            "Hard cap on detection count. Sources beyond the cap are dropped "
            "by ascending score (lowest-scoring first)."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description="Output FITS stem. Defaults to '{starless_stem}_selective_reblend'.",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_fits_color(image_path: Path) -> tuple[np.ndarray, bool]:
    """Load FITS, return ((C,H,W) float32, is_color)."""
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        return data, True
    if data.ndim == 3 and data.shape[2] == 3:
        return np.moveaxis(data, -1, 0), True
    return data.squeeze()[np.newaxis, :, :], False


def _luminance(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 1:
        return data[0]
    return 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]


def _normalize_mask_to_2d(mask: np.ndarray) -> np.ndarray:
    """Collapse a mask FITS to (H, W) float32 by per-pixel max across channels."""
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 3:
            return mask.max(axis=0).astype(np.float32)
        if mask.shape[2] == 3:
            return mask.max(axis=2).astype(np.float32)
        raise ValueError(f"Unexpected mask shape {mask.shape}")
    return mask.astype(np.float32)


def _saturation_at(rgb: np.ndarray, y: int, x: int) -> float:
    r, g, b = float(rgb[0, y, x]), float(rgb[1, y, x]), float(rgb[2, y, x])
    mx = max(r, g, b)
    if mx <= 0.0:
        return 0.0
    mn = min(r, g, b)
    return (mx - mn) / mx


def _annulus_saturation(rgb: np.ndarray, y: int, x: int, r_in: float, r_out: float) -> float:
    H, W = rgb.shape[1], rgb.shape[2]
    r_o = int(np.ceil(r_out))
    y_lo, y_hi = max(0, y - r_o), min(H, y + r_o + 1)
    x_lo, x_hi = max(0, x - r_o), min(W, x + r_o + 1)
    if y_hi <= y_lo or x_hi <= x_lo:
        return 0.0
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    rr2 = (yy - y) ** 2 + (xx - x) ** 2
    in_ann = (rr2 >= r_in ** 2) & (rr2 <= r_out ** 2)
    if not np.any(in_ann):
        return 0.0
    patch = rgb[:, y_lo:y_hi, x_lo:x_hi]
    mx = patch.max(axis=0)
    mn = patch.min(axis=0)
    sat = np.where(mx > 0.0, (mx - mn) / np.where(mx > 0.0, mx, 1.0), 0.0)
    return float(np.mean(sat[in_ann]))


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return ranks / (len(values) - 1)


def _compose_score(peaks: np.ndarray, chroma: np.ndarray, mode: str) -> np.ndarray:
    peak_norm = peaks / (peaks.max() + 1e-9)
    chroma_norm = chroma / (chroma.max() + 1e-9) if chroma.max() > 0 else np.zeros_like(chroma)
    if mode == "brightness_priority":
        return peak_norm * (1.0 + 0.3 * chroma_norm)
    if mode == "color_priority":
        return chroma_norm * (1.0 + 0.3 * peak_norm)
    if mode == "balanced":
        return 0.5 * (_percentile_rank(peaks) + _percentile_rank(chroma))
    raise ValueError(f"Unknown mode: {mode}")


def _detect_on_mask(
    mask_lum: np.ndarray,
    threshold_sigma: float,
    fwhm_guess: float,
    min_sep: float,
):
    bg_noise = float(mad_std(mask_lum))
    if bg_noise <= 0:
        return None, 0.0, 0.0
    bg_level = float(np.median(mask_lum))
    threshold = threshold_sigma * bg_noise
    from photutils.detection import IRAFStarFinder
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = IRAFStarFinder(
            threshold=threshold, fwhm=fwhm_guess, minsep_fwhm=min_sep,
        )
        sources = finder(mask_lum - bg_level)
    return sources, bg_noise, bg_level


def _build_weight_map(
    shape: tuple[int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    fwhms: np.ndarray,
    keep_mask: np.ndarray,
    suppress_strength: float,
    core_radius_factor: float,
    feather_sigma_px: float,
) -> np.ndarray:
    """Paint per-source disks; kept sources get W=1.0, others suppress_strength."""
    H, W_dim = shape
    W = np.full(shape, suppress_strength, dtype=np.float32)
    for i in range(len(xs)):
        if not keep_mask[i]:
            continue
        cy, cx = int(round(float(ys[i]))), int(round(float(xs[i])))
        r = max(1, int(round(float(fwhms[i]) * core_radius_factor)))
        y_lo, y_hi = max(0, cy - r), min(H, cy + r + 1)
        x_lo, x_hi = max(0, cx - r), min(W_dim, cx + r + 1)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        in_disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        sub = W[y_lo:y_hi, x_lo:x_hi]
        sub[in_disk] = 1.0
        W[y_lo:y_hi, x_lo:x_hi] = sub
    if feather_sigma_px > 0:
        W = gaussian(W, sigma=feather_sigma_px)
    return np.clip(W, 0.0, 1.0).astype(np.float32)


# ── LangChain tool ─────────────────────────────────────────────────────────────


@tool(args_schema=SelectiveStarReblendInput)
def selective_star_reblend(
    mode: str = "brightness_priority",
    keep_fraction: float = 0.1,
    suppress_strength: float = 0.3,
    core_radius_factor: float = 1.5,
    feather_sigma_px: float = 2.0,
    mask_dilation_px: int = 0,
    confine_to_region_mask: bool = False,
    threshold_sigma: float = 5.0,
    fwhm_guess: float = 3.0,
    min_separation_fwhm: float = 2.0,
    max_sources: int = 5000,
    output_stem: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Recombine starless + star_mask using a per-source weight map W in [0,1].

    Detection runs on the star_mask. Each source is scored per `mode`; the
    top `keep_fraction` by score receive W=1.0 (full restoration). Other
    sources receive W=`suppress_strength`. Per-source weights are painted
    as feathered disks of radius `fwhm * core_radius_factor` and Gaussian-
    blurred by `feather_sigma_px`. Optional `mask_dilation_px` dilates the
    StarNet mask before blending. Optional `confine_to_region_mask` reads
    paths.latest_mask and applies the weight rules only inside it (W=1.0
    outside, full restoration).

    Output: starless + dilated_star_mask * W. New FITS path becomes
    paths.current_image; pre-call current_image is preserved at
    paths.previous_image. Reads paths.starless_image and paths.star_mask
    from state — call star_removal first.
    """
    working_dir = state["dataset"]["working_dir"]
    starless_p = state["paths"].get("starless_image")
    mask_p = state["paths"].get("star_mask")
    current_p = state["paths"].get("current_image")

    if not starless_p or not Path(starless_p).exists():
        raise FileNotFoundError(
            "selective_star_reblend: paths.starless_image is missing or the "
            "file does not exist on disk. Call star_removal first to produce "
            "the starless + star_mask pair."
        )
    if not mask_p or not Path(mask_p).exists():
        raise FileNotFoundError(
            "selective_star_reblend: paths.star_mask is missing or the file "
            "does not exist on disk. Call star_removal first to produce "
            "the starless + star_mask pair."
        )

    # State authority on image_space — see Metadata.image_space and CLAUDE.md.
    incoming_image_space = state["metadata"].get("image_space")
    if incoming_image_space not in ("linear", "display"):
        raise RuntimeError(
            "selective_star_reblend: state.metadata.image_space is missing or "
            f"invalid (got {incoming_image_space!r}). Refusing to guess — "
            "restart from a fresh checkpoint."
        )

    # Region-mask zoning.
    region_mask_2d: np.ndarray | None = None
    if confine_to_region_mask:
        region_p = state["paths"].get("latest_mask")
        if not region_p or not Path(region_p).exists():
            raise FileNotFoundError(
                "selective_star_reblend: confine_to_region_mask=True but "
                "paths.latest_mask is missing. Call create_mask first to "
                "produce the region mask."
            )
        with astropy_fits.open(region_p) as hdul:
            rm = hdul[0].data.astype(np.float32)
        if rm.max() > 1.0:
            rm = rm / rm.max()
        rm = np.squeeze(rm)
        if rm.ndim == 3:
            rm = rm.max(axis=0) if rm.shape[0] == 3 else rm.max(axis=2)
        region_mask_2d = (rm > 0.5).astype(np.float32)

    # Load starless and mask.
    starless, starless_is_color = _load_fits_color(Path(starless_p))
    with astropy_fits.open(mask_p) as hdul:
        mask_raw = hdul[0].data.astype(np.float32)

    if starless.max() > 1.0:
        starless = starless / starless.max()
    if mask_raw.max() > 1.0:
        mask_raw = mask_raw / mask_raw.max()

    # Mask in (C, H, W) for blending; mask luminance for detection.
    mask_squeezed = np.squeeze(mask_raw)
    if mask_squeezed.ndim == 3:
        if mask_squeezed.shape[0] == 3:
            mask_chw = mask_squeezed
        elif mask_squeezed.shape[2] == 3:
            mask_chw = np.moveaxis(mask_squeezed, -1, 0)
        else:
            raise ValueError(f"Unexpected mask shape {mask_squeezed.shape}")
    else:
        mask_chw = mask_squeezed[np.newaxis, :, :]

    # Broadcast mono mask to 3 channels if starless is color.
    if starless_is_color and mask_chw.shape[0] == 1:
        mask_chw = np.broadcast_to(mask_chw, (3, *mask_chw.shape[1:])).copy()
    if not starless_is_color and mask_chw.shape[0] == 3:
        mask_chw = _luminance(mask_chw)[np.newaxis, :, :]

    if mask_chw.shape[1:] != starless.shape[1:]:
        raise ValueError(
            f"starless ({starless.shape[1:]}) and star_mask "
            f"({mask_chw.shape[1:]}) shapes disagree."
        )
    mask_lum = _luminance(mask_chw)

    if region_mask_2d is not None and region_mask_2d.shape != mask_lum.shape:
        raise ValueError(
            f"region mask ({region_mask_2d.shape}) and image "
            f"({mask_lum.shape}) shapes disagree."
        )

    # Detect on mask luminance.
    sources, bg_noise, bg_level = _detect_on_mask(
        mask_lum, threshold_sigma, fwhm_guess, min_separation_fwhm,
    )

    if sources is None or len(sources) == 0:
        raise RuntimeError(
            "selective_star_reblend: no sources detected on the star_mask "
            f"at threshold_sigma={threshold_sigma}, fwhm_guess={fwhm_guess}. "
            "Lower threshold_sigma, or check that the mask is non-empty."
        )

    xs = np.array(sources["xcentroid"], dtype=np.float32)
    ys = np.array(sources["ycentroid"], dtype=np.float32)
    peaks = np.array(sources["peak"], dtype=np.float32)
    fwhms = np.array(sources["fwhm"], dtype=np.float32)

    # Chroma (annulus) sampled from the mask itself — the mask carries the
    # star color. Falls back to 0 for mono masks.
    if mask_chw.shape[0] == 3:
        chroma_list: list[float] = []
        for i in range(len(peaks)):
            yi = max(0, min(mask_chw.shape[1] - 1, int(round(float(ys[i])))))
            xi = max(0, min(mask_chw.shape[2] - 1, int(round(float(xs[i])))))
            fi = float(fwhms[i])
            chroma_list.append(_annulus_saturation(mask_chw, yi, xi, fi, 2.0 * fi))
        chroma = np.array(chroma_list, dtype=np.float32)
    else:
        chroma = np.zeros_like(peaks)

    score = _compose_score(peaks, chroma, mode)

    # Cap by score.
    if len(score) > max_sources:
        keep_idx = np.argsort(score)[-max_sources:]
        xs, ys = xs[keep_idx], ys[keep_idx]
        peaks, fwhms, chroma, score = peaks[keep_idx], fwhms[keep_idx], chroma[keep_idx], score[keep_idx]

    # Pick the keep set.
    n_keep = max(0, int(round(keep_fraction * len(score))))
    keep_threshold_idx = np.argsort(score)[-n_keep:] if n_keep > 0 else np.array([], dtype=int)
    keep_mask = np.zeros(len(score), dtype=bool)
    if n_keep > 0:
        keep_mask[keep_threshold_idx] = True

    # Build W on the image grid.
    H, W_dim = mask_lum.shape
    W = _build_weight_map(
        (H, W_dim), xs, ys, fwhms, keep_mask,
        suppress_strength, core_radius_factor, feather_sigma_px,
    )

    # Confine W to region (outside region: full restoration).
    if region_mask_2d is not None:
        W = np.where(region_mask_2d > 0.5, W, 1.0).astype(np.float32)

    # Optional mask dilation.
    if mask_dilation_px > 0:
        # Build a binary support, dilate, then re-apply soft mask values
        # only on the dilated support.
        support = mask_lum > (5.0 * bg_noise + bg_level)
        dilated = binary_dilation(support, disk(mask_dilation_px))
        # Spread the soft mask values into newly dilated pixels by nearest
        # value via a lightweight Gaussian blur of the soft mask, then
        # apply only on the dilated support.
        spread = gaussian(mask_chw, sigma=float(mask_dilation_px), channel_axis=0)
        out_mask = np.where(
            dilated[np.newaxis, :, :], np.maximum(mask_chw, spread), mask_chw
        ).astype(np.float32)
    else:
        out_mask = mask_chw

    # Blend.
    result = starless.astype(np.float32) + out_mask * W[np.newaxis, :, :]
    result = np.clip(result, 0.0, 1.0)

    # Save.
    starless_stem = Path(starless_p).stem
    out_stem = output_stem or f"{starless_stem}_selective_reblend"
    out_path = Path(working_dir) / f"{out_stem}.fits"
    if not starless_is_color:
        result_to_write = result[0]
    else:
        result_to_write = result
    astropy_fits.HDUList(
        [astropy_fits.PrimaryHDU(data=result_to_write)]
    ).writeto(out_path, overwrite=True)

    n_kept = int(keep_mask.sum())
    n_total = int(len(keep_mask))
    payload = {
        "output_path": str(out_path),
        "mode": mode,
        "keep_fraction": keep_fraction,
        "suppress_strength": suppress_strength,
        "sources_total": n_total,
        "sources_kept": n_kept,
        "sources_suppressed": n_total - n_kept,
        "confine_to_region_mask": confine_to_region_mask,
        "score_p10_p50_p90": [
            float(np.percentile(score, 10)),
            float(np.percentile(score, 50)),
            float(np.percentile(score, 90)),
        ],
        "bg_noise_mad": bg_noise,
        "image_space": incoming_image_space,
    }

    prev_current = current_p
    return Command(update={
        "paths": {
            "current_image": str(out_path),
            "previous_image": prev_current,
        },
        "metadata": {"image_space": incoming_image_space},
        "messages": [ToolMessage(
            content=json.dumps(payload, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    })
