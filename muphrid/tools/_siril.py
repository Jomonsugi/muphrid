"""
Siril script engine — the single execution path for all siril-cli invocations.

Every tool that calls Siril uses run_siril_script(). Nothing else invokes
siril-cli directly. This keeps subprocess handling, error parsing, and
logging in one place.

Siril script format (.ssf):
    requires 1.4.0
    cd /absolute/path/to/working/dir
    load image.fit
    # ... processing commands ...
    savefits result.fit
    close

Invocation:
    siril-cli -d <working_dir> -s <script_path>
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits as astropy_fits

from muphrid.config import load_settings


def siril_script_path(path: Path | str, working_dir: Path | str) -> str:
    """
    Return a whitespace-free reference to ``path`` for use inside a Siril
    script command line.

    Siril 1.4's script tokenizer splits arguments on raw whitespace BEFORE
    any quote handling — double-quoting does not help. A path such as::

        -bias="/Users/x/COCOON NEBULA/master_bias.fit"

    is split into two tokens at the space, leaving the stray quote in the
    first one. Siril then tries to open ``/Users/x/COCOON`` and fails with::

        log: "/Users/x/COCOON.[any_allowed_extension] not found

    To make Siril commands work regardless of how the user named their
    dataset folder, this helper guarantees Siril only ever sees short,
    whitespace-free references:

      - If ``path`` is under ``working_dir`` and the relative form has no
        whitespace, the relative path is returned. Siril resolves it
        against the ``-d`` working directory (which is passed as a single
        argv token at the subprocess layer, so spaces there are fine).
      - Otherwise a symlink is created in ``working_dir`` under a
        whitespace-free name, and the basename is returned. The symlink
        is idempotent — repeat calls reuse any existing link that already
        points at ``path``.

    The symlink fallback is safe: muphrid run folders are per-session,
    so leftover symlinks die with the run folder.
    """
    import hashlib
    import re

    src = Path(path).resolve()
    wd = Path(working_dir).resolve()

    # Case 1: path is inside working_dir and the relative form is clean.
    try:
        rel_str = str(src.relative_to(wd))
        if not re.search(r"\s", rel_str):
            return rel_str
    except ValueError:
        pass  # path lives outside working_dir — fall through to symlink

    # Case 2: symlink into working_dir under a whitespace-free basename.
    clean_name = re.sub(r"\s+", "_", src.name) or "file"

    link = wd / clean_name

    # Disambiguate on collision with an unrelated file/link at that name.
    already_points_here = (
        link.is_symlink() and link.resolve() == src
    )
    if (link.exists() or link.is_symlink()) and not already_points_here:
        digest = hashlib.sha1(str(src).encode()).hexdigest()[:8]
        clean_name = f"{digest}_{clean_name}"
        link = wd / clean_name

    if link.is_symlink():
        if link.resolve() != src:
            link.unlink()
            link.symlink_to(src)
    elif not link.exists():
        wd.mkdir(parents=True, exist_ok=True)
        link.symlink_to(src)

    return clean_name


def fits_nlayers(fits_path: Path | str) -> int:
    """Return the number of color channels in a FITS image.

    Returns 1 for monochrome (NAXIS<3 or NAXIS3=1), 3 for RGB (NAXIS3=3).
    """
    with astropy_fits.open(str(fits_path)) as hdul:
        hdr = hdul[0].header
        naxis = int(hdr.get("NAXIS", 0))
        if naxis < 3:
            return 1
        return int(hdr.get("NAXIS3", 1))


def fits_has_nan(fits_path: Path | str) -> bool:
    """Return True if the FITS image data contains any NaN or Inf values."""
    import numpy as np
    with astropy_fits.open(str(fits_path)) as hdul:
        data = hdul[0].data
        return bool(np.any(~np.isfinite(data)))


# ── Result and error types ─────────────────────────────────────────────────────

@dataclass
class SirilResult:
    stdout: str
    stderr: str
    exit_code: int
    script: str                         # the .ssf content that was executed
    working_dir: str
    parsed: dict = field(default_factory=dict)  # structured values extracted from stdout


class SirilError(RuntimeError):
    """
    Raised when siril-cli exits with a non-zero code or stdout contains
    a known error pattern.
    """
    def __init__(self, message: str, result: SirilResult) -> None:
        super().__init__(message)
        self.result = result


# ── Script patterns ────────────────────────────────────────────────────────────
# Regexes applied to stdout after each script run to extract structured values.
# Each entry: (key, compiled_pattern, group_index_or_name)

_STDOUT_PATTERNS: list[tuple[str, re.Pattern, str | int]] = [
    ("background_noise",  re.compile(r"Background noise level:\s*([\d.e+-]+)"), 1),
    ("background_mean",   re.compile(r"Background mean:\s*([\d.e+-]+)"),        1),
    ("fwhm",              re.compile(r"FWHM:\s*([\d.]+)"),                      1),
    ("star_count",        re.compile(r"(\d+)\s+stars?\s+detected",
                                     re.IGNORECASE),                            1),
    ("snr",               re.compile(r"SNR[:\s]+([\d.]+)"),                     1),
    ("rejected_frames",   re.compile(r"(\d+)\s+frame[s]?\s+rejected",
                                     re.IGNORECASE),                            1),
    ("accepted_frames",   re.compile(r"(\d+)\s+frame[s]?\s+stacked",
                                     re.IGNORECASE),                            1),
    ("output_path",       re.compile(r"Saving FITS image:\s*(.+\.fit[s]?)"),    1),
]

_SUCCESS_MARKER = "Script execution finished successfully"

# Patterns that indicate a real script failure.  Applied only when the script
# did NOT report success (the success marker gates pattern-based detection).
# Verified against Siril 1.4.2 runtime output — "Error in line N" is the
# definitive per-command failure message; "Script execution failed" is the
# global failure message.
_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(r"Error in line \d+",            re.IGNORECASE),
    re.compile(r"Script execution failed",      re.IGNORECASE),
    re.compile(r"command not found",            re.IGNORECASE),
    re.compile(r"No such file or directory",    re.IGNORECASE),
]


def _parse_stdout(stdout: str) -> dict:
    parsed: dict = {}
    for key, pattern, group in _STDOUT_PATTERNS:
        match = pattern.search(stdout)
        if match:
            try:
                parsed[key] = float(match.group(group))  # type: ignore[arg-type]
            except (ValueError, TypeError):
                parsed[key] = match.group(group)         # keep as string if not numeric
    return parsed


def _check_for_errors(result: SirilResult, exit_code: int) -> None:
    if exit_code != 0:
        raise SirilError(
            f"siril-cli exited with code {exit_code}.\n"
            f"stderr: {result.stderr.strip() or '(empty)'}\n"
            f"stdout tail: {result.stdout[-500:].strip() or '(empty)'}",
            result,
        )

    # If Siril itself reports success, trust it — do not second-guess with
    # pattern matching.  The old broad "^\s*ERROR\b" regex was triggering on
    # informational log lines that contained the word ERROR even when the
    # script completed without any command failures.
    if _SUCCESS_MARKER in result.stdout:
        return

    combined = result.stdout + result.stderr
    for pattern in _ERROR_PATTERNS:
        if pattern.search(combined):
            raise SirilError(
                f"Siril reported an error in its output.\n"
                f"Matched pattern: {pattern.pattern!r}\n"
                f"stdout tail: {result.stdout[-500:].strip()}",
                result,
            )


# ── Script builder helpers ─────────────────────────────────────────────────────

def build_script(commands: list[str], requires: str = "1.4.0") -> str:
    """
    Wrap a list of Siril commands into a complete .ssf script.

    Args:
        commands: Siril commands in order. Do NOT include 'requires' or 'close' —
                  these are added automatically.
        requires: Minimum Siril version required by this script.

    Returns:
        Complete .ssf script content as a string.
    """
    lines = [f"requires {requires}", ""] + commands + ["", "close"]
    return "\n".join(lines) + "\n"


# ── Main execution function ────────────────────────────────────────────────────

def run_siril_script(
    commands: list[str],
    working_dir: str,
    *,
    requires: str = "1.4.0",
    timeout: int = 600,
    siril_bin: str | None = None,
) -> SirilResult:
    """
    Execute a sequence of Siril commands and return structured results.

    Writes the commands to a temporary .ssf file, invokes siril-cli, captures
    output, parses known metrics from stdout, and raises SirilError on failure.

    Args:
        commands:    Siril script commands to execute (without 'requires'/'close').
        working_dir: Absolute path passed to siril-cli via -d flag. Siril treats
                     this as the working directory for all relative file paths.
        requires:    Minimum Siril version declared in the script header.
        timeout:     Subprocess timeout in seconds. Default 600s (10 min) for
                     long operations like stacking.
        siril_bin:   Override the siril-cli binary path. Falls back to SIRIL_BIN
                     env var, then 'siril-cli'.

    Returns:
        SirilResult with stdout, stderr, the script content, and parsed metrics.

    Raises:
        SirilError:  Non-zero exit code or error pattern found in output.
        FileNotFoundError: working_dir does not exist.
        subprocess.TimeoutExpired: Script exceeded timeout.
    """
    working_path = Path(working_dir)
    if not working_path.exists():
        raise FileNotFoundError(
            f"working_dir does not exist: {working_dir}"
        )

    if siril_bin is None:
        siril_bin = os.environ.get("SIRIL_BIN", "siril-cli")

    script_content = build_script(commands, requires=requires)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ssf",
        prefix="muphrid_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(script_content)
        script_path = f.name

    try:
        proc = subprocess.run(
            [siril_bin, "-d", working_dir, "-s", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)

    result = SirilResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
        script=script_content,
        working_dir=working_dir,
        parsed=_parse_stdout(proc.stdout),
    )

    _check_for_errors(result, proc.returncode)
    return result
