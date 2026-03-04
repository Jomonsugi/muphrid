"""
T27 — multiscale_process

Decompose the image into discrete spatial frequency scales via the B3-spline
à trous (undecimated, shift-invariant) wavelet transform, apply independent
operations per scale, then reconstruct. This is the open-source equivalent of
PixInsight's Multiscale Linear Transform (MLT).

IMPORTANT: PyWavelets has no 'b3' wavelet. The B3-spline à trous transform is
implemented directly using scipy.ndimage.convolve1d with the kernel:
  [1/16, 4/16, 6/16, 4/16, 1/16]
applied separably at each scale with stride=2^i (à trous = "with holes").
This is isotropic (no orientation bias), artifact-free, and the same
decomposition used by PixInsight MLT and NoiseXterminator.

Per-scale operations:
  sharpen   — multiply detail coefficients by weight (>1 boosts, <1 suppresses)
  denoise   — soft-threshold coefficients using MAD-normalized threshold
  suppress  — zero the coefficients (remove this scale's contribution entirely)
  passthrough — leave coefficients unchanged

Backend: Pure Python — scipy.ndimage + Astropy. No Siril invocation.

Standard recipes (from spec §5 T27):
  Sharpening (on starless image with nebula luminance mask from T25):
    scale 1: suppress, scale 2: sharpen weight=1.3, scale 3: sharpen weight=1.15,
    scale 4+: passthrough

  Noise reduction:
    scale 1: denoise sigma=0.5, scale 2: denoise sigma=0.2,
    scale 3+: passthrough
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits as astropy_fits
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from scipy.ndimage import convolve1d


# ── B3-spline à trous transform ────────────────────────────────────────────────

B3_SPLINE_KERNEL = np.array([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16], dtype=np.float32)


def b3_atrous_decompose(data: np.ndarray, num_scales: int) -> list[np.ndarray]:
    """
    Compute the B3-spline à trous wavelet decomposition.

    Returns a list of length num_scales + 1:
      [detail_scale_1, detail_scale_2, ..., detail_scale_N, approximation_residual]

    Reconstruction: sum all elements in the returned list.
    """
    layers = []
    approx = data.copy().astype(np.float32)

    for i in range(num_scales):
        stride = 2 ** i
        # Build sparse 1D kernel with interleaved zeros (holes)
        # Length: 4 * stride + 1 (4 gaps between 5 kernel values)
        k_len = 4 * stride + 1
        sparse_kernel = np.zeros(k_len, dtype=np.float32)
        for j, c in enumerate(B3_SPLINE_KERNEL):
            sparse_kernel[j * stride] = c

        # Apply separable 2D convolution (horizontal then vertical)
        smooth_h = convolve1d(approx, sparse_kernel, axis=1, mode="reflect")
        smooth = convolve1d(smooth_h, sparse_kernel, axis=0, mode="reflect")

        detail = approx - smooth
        layers.append(detail)
        approx = smooth

    layers.append(approx)  # Final approximation residual
    return layers


def b3_atrous_reconstruct(layers: list[np.ndarray]) -> np.ndarray:
    """Sum all B3 atrous layers to reconstruct the image."""
    return sum(layers).astype(np.float32)


def _soft_threshold(coeffs: np.ndarray, sigma_factor: float) -> np.ndarray:
    """
    MAD-normalized soft thresholding for wavelet denoising.

    Threshold = sigma_factor * median(|coeffs|) / 0.6745
    Applied as: sign(c) * max(|c| - threshold, 0)
    """
    mad = float(np.median(np.abs(coeffs)))
    threshold = sigma_factor * mad / 0.6745
    sign = np.sign(coeffs)
    return sign * np.maximum(np.abs(coeffs) - threshold, 0.0)


# ── Pydantic input schema ──────────────────────────────────────────────────────

class ScaleOperation(BaseModel):
    scale: int = Field(
        description=(
            "Scale index (1-based). Scale 1 = finest detail (2–4px features, "
            "noise and grain). Higher scales = coarser features."
        )
    )
    operation: str = Field(
        description=(
            "Operation to apply at this scale:\n"
            "  'sharpen': multiply coefficients by weight (>1 = boost, <1 = reduce).\n"
            "  'denoise': soft-threshold coefficients with MAD-normalized threshold.\n"
            "  'suppress': zero all coefficients (remove this scale entirely).\n"
            "  'passthrough': leave coefficients unchanged."
        )
    )
    weight: float = Field(
        default=1.0,
        description=(
            "Multiplier for 'sharpen' operation. "
            "1.3 = 30% boost (standard). 0.5 = 50% suppression. "
            "Ignored for other operations."
        ),
    )
    denoise_sigma: float | None = Field(
        default=None,
        description=(
            "Sigma factor for 'denoise' operation. "
            "Threshold = sigma * median(|coeffs|) / 0.6745. "
            "0.5 = moderate (scale 1). 0.2 = gentle (scale 2). "
            "Required when operation='denoise'."
        ),
    )


class MultiscaleProcessInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the working directory."
    )
    image_path: str = Field(
        description="Absolute path to the FITS image to process."
    )
    num_scales: int = Field(
        default=5,
        description=(
            "Number of wavelet scales to decompose into (1–6). "
            "Scale 1: 2–4px (noise). Scale 2: 4–8px (fine detail). "
            "Scale 3: 8–16px (shells, lanes). Scale 4: 16–32px (galaxy arms). "
            "Scale 5+: residual (large-scale background). "
            "5 scales covers most astrophotography use cases."
        ),
    )
    scale_operations: list[ScaleOperation] = Field(
        description=(
            "List of per-scale operations. Scales not listed default to passthrough. "
            "See spec §5 T27 for standard sharpening and denoising recipes."
        )
    )
    mask_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a float32 FITS mask from T25. If provided, the "
            "processed result is blended with the original via the mask: "
            "'result * mask + original * (1 - mask)'. "
            "Use an inverted_luminance mask for noise reduction to avoid "
            "over-smoothing nebula structure."
        ),
    )
    per_channel: bool = Field(
        default=False,
        description=(
            "If False (default): convert to luminance, process, recombine. "
            "Preserves color ratios and prevents chromatic artifacts. "
            "If True: process each RGB channel independently. "
            "Use only when channels have very different noise characteristics."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description="Output FITS stem. Defaults to '{source_stem}_mlt'.",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_fits(image_path: Path) -> tuple[np.ndarray, bool]:
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.max() > 1.0:
        data = data / data.max()

    if data.ndim == 3 and data.shape[0] == 3:
        return data, True
    elif data.ndim == 3 and data.shape[2] == 3:
        return np.moveaxis(data, -1, 0).astype(np.float32), True
    else:
        mono = data.squeeze().astype(np.float32)
        return mono[np.newaxis, :, :], False


def _load_mask(mask_path: Path, target_shape: tuple) -> np.ndarray:
    with astropy_fits.open(mask_path) as hdul:
        m = hdul[0].data.astype(np.float32).squeeze()
    if m.max() > 1.0:
        m = m / m.max()
    return np.clip(m, 0.0, 1.0)


def _build_op_map(scale_operations: list[ScaleOperation], num_scales: int) -> dict:
    """Build {scale_index: ScaleOperation} dict with passthrough defaults."""
    op_map = {i: ScaleOperation(scale=i, operation="passthrough") for i in range(1, num_scales + 1)}
    for op in scale_operations:
        if 1 <= op.scale <= num_scales:
            op_map[op.scale] = op
    return op_map


def _apply_operation(
    detail: np.ndarray,
    scale_op: ScaleOperation,
) -> tuple[np.ndarray, float, float]:
    """
    Apply a single scale operation to detail coefficients.
    Returns (modified_detail, energy_before, energy_after).
    """
    energy_before = float(np.mean(detail ** 2))

    op = scale_op.operation.lower()
    if op == "sharpen":
        result = detail * scale_op.weight
    elif op == "denoise":
        sigma = scale_op.denoise_sigma
        if sigma is None or sigma <= 0:
            raise ValueError(
                f"Scale {scale_op.scale}: denoise_sigma must be > 0 "
                "(e.g. 0.5 for scale 1, 0.2 for scale 2)."
            )
        result = _soft_threshold(detail, sigma)
    elif op == "suppress":
        result = np.zeros_like(detail)
    elif op == "passthrough":
        result = detail
    else:
        raise ValueError(
            f"Unknown operation '{scale_op.operation}'. "
            "Valid: sharpen, denoise, suppress, passthrough."
        )

    energy_after = float(np.mean(result ** 2))
    return result.astype(np.float32), energy_before, energy_after


def _process_plane(
    plane: np.ndarray,
    num_scales: int,
    op_map: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Decompose, apply operations, reconstruct a single 2D plane."""
    layers = b3_atrous_decompose(plane, num_scales)

    per_scale_stats = []
    for scale_idx in range(1, num_scales + 1):
        layer_i = scale_idx - 1
        op = op_map[scale_idx]
        modified, e_before, e_after = _apply_operation(layers[layer_i], op)
        layers[layer_i] = modified
        per_scale_stats.append({
            "scale": scale_idx,
            "operation": op.operation,
            "coefficient_energy_before": round(e_before, 8),
            "coefficient_energy_after": round(e_after, 8),
        })

    reconstructed = b3_atrous_reconstruct(layers)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    return reconstructed, per_scale_stats


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=MultiscaleProcessInput)
def multiscale_process(
    working_dir: str,
    image_path: str,
    num_scales: int = 5,
    scale_operations: list[ScaleOperation] | None = None,
    mask_path: str | None = None,
    per_channel: bool = False,
    output_stem: str | None = None,
) -> dict:
    """
    Decompose the image into spatial frequency scales via the B3-spline à trous
    wavelet transform, apply independent operations per scale, then reconstruct.

    This is the open-source equivalent of PixInsight's Multiscale Linear
    Transform (MLT). It provides surgical control unavailable in Siril's
    wavelet+wrecons: sharpen scale 2 while suppressing scale 1 noise, leave
    large-scale structure untouched, and confine everything to a masked region.

    Standard sharpening recipe (starless image, with nebula mask from T25):
      scale 1: suppress        (remove noise before sharpening)
      scale 2: sharpen 1.3     (fine filaments, PSF wings)
      scale 3: sharpen 1.15    (shells, dust lane edges)
      scale 4: passthrough
      scale 5: passthrough

    Standard noise reduction recipe (linear or post-stretch, no mask needed):
      scale 1: denoise sigma=0.5
      scale 2: denoise sigma=0.2
      scale 3–5: passthrough

    Implementation note: PyWavelets has no 'b3' wavelet. The B3-spline à trous
    transform is implemented directly using scipy.ndimage.convolve1d with kernel
    [1/16, 4/16, 6/16, 4/16, 1/16] and inter-scale holes (stride=2^i).
    """
    if scale_operations is None:
        scale_operations = []

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    data, is_color = _load_fits(img_path)
    original = data.copy()
    op_map = _build_op_map(scale_operations, num_scales)

    per_scale_stats: list[dict] = []

    if not per_channel:
        # Process luminance only, recombine
        lum = (0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]).astype(np.float32)
        processed_lum, stats = _process_plane(lum, num_scales, op_map)
        per_scale_stats = stats

        # Recombine: scale each channel by the luminance ratio
        lum_safe = np.where(lum > 1e-6, lum, 1e-6)
        ratio = processed_lum / lum_safe
        output = np.stack([
            np.clip(data[0] * ratio, 0.0, 1.0),
            np.clip(data[1] * ratio, 0.0, 1.0),
            np.clip(data[2] * ratio, 0.0, 1.0),
        ], axis=0)
    else:
        # Process each channel independently
        output_channels = []
        for ch_idx in range(data.shape[0]):
            processed_ch, stats = _process_plane(data[ch_idx], num_scales, op_map)
            output_channels.append(processed_ch)
            if ch_idx == 0:
                per_scale_stats = stats  # Report stats from first channel
        output = np.stack(output_channels, axis=0)

    # Optional mask blending
    if mask_path and Path(mask_path).exists():
        mask = _load_mask(Path(mask_path), lum.shape if not per_channel else data.shape[1:])
        # Broadcast mask to (3, H, W)
        mask_3d = mask[np.newaxis, :, :]
        output = output * mask_3d + original * (1.0 - mask_3d)
        output = np.clip(output, 0.0, 1.0)

    # Save result
    out_stem = output_stem or f"{img_path.stem}_mlt"
    out_path = Path(working_dir) / f"{out_stem}.fits"
    hdu = astropy_fits.PrimaryHDU(data=output)
    astropy_fits.HDUList([hdu]).writeto(out_path, overwrite=True)

    return {
        "processed_image_path": str(out_path),
        "per_scale_stats": per_scale_stats,
    }
