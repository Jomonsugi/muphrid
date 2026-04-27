# Patch — Polynomial gradient fallback + regression advisory

**File:** `/Users/micahshanks/Dev/muphrid/muphrid/tools/linear/t09_gradient.py`

## Why

In the M20 run, GraXpert turned a flat background (`gradient_magnitude = 0.012`) into a strongly graded one (0.462 at s=0.6, 0.428 at s=0.8). The agent had no fallback — GraXpert is the only backend — so it committed the regression and advanced. Giving the tool a polynomial fallback (Siril's built-in `background -gradient`) restores the classic DBE/ABE-style option that every pro workflow has available, and returning a structured regression warning forces the feedback loop to be real, not advisory.

## Shape of the change

Extend `RemoveGradientInput` with a `method` discriminator and a polynomial branch; compute a regression check against the pre-image and return it in a structured field.

```python
from typing import Literal

class PolynomialBGEOptions(BaseModel):
    degree: int = Field(
        default=2,
        ge=1, le=4,
        description=(
            "Polynomial degree of the 2D background model. "
            "1 = planar (bias + linear tilt). Safe on compact targets. "
            "2 = quadratic (most light-pollution gradients). Good default. "
            "3 = cubic (complex gradients, risk of over-fitting extended nebulae). "
            "4 = quartic (rarely justified; over-fits easily). "
            "Increase only if the residual after degree=2 is still visibly graded."
        ),
    )
    samples_per_line: int = Field(
        default=20,
        ge=5, le=60,
        description="Siril -samples parameter; higher = finer sampling grid.",
    )
    tolerance: float = Field(
        default=1.0,
        description="Siril -tolerance parameter for sample rejection.",
    )


class RemoveGradientInput(BaseModel):
    method: Literal["graxpert", "polynomial", "siril_auto"] = Field(
        default="graxpert",
        description=(
            "graxpert: AI background extraction. Best on irregular gradients; "
            "can over-fit extended targets (check the regression_warning in the "
            "result and switch methods if triggered).\n"
            "polynomial: Siril `background -gradient` with degree N. Deterministic, "
            "safe on compact targets, preferred fallback when GraXpert regresses "
            "or when the target fills a large fraction of the frame.\n"
            "siril_auto: Siril's automatic background extraction. Uses morphological "
            "opening plus polynomial fit — an ABE-style approach. Fast, no AI."
        ),
    )
    graxpert_options: GraXpertBGEOptions | None = Field(default_factory=GraXpertBGEOptions)
    polynomial_options: PolynomialBGEOptions | None = Field(default=None)
    chain: bool = Field(default=False, description="...")
```

In the tool body, branch on `method`:

```python
if method == "polynomial":
    # Build Siril script:
    #   load <stem>
    #   background -gradient -samples=N -tolerance=T -degree=D
    #   save <output_stem>
    commands = [
        f"load {stem}",
        (
            f"background -gradient "
            f"-samples={polynomial_options.samples_per_line} "
            f"-tolerance={polynomial_options.tolerance} "
            f"-degree={polynomial_options.degree}"
        ),
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=120)

elif method == "siril_auto":
    commands = [
        f"load {stem}",
        "background -auto",
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=120)

else:  # graxpert (existing path)
    _run_graxpert_bge(...)
```

## Regression advisory

After the operation, compare key metrics pre- and post-tool. If `gradient_magnitude` got worse OR `signal_coverage_pct` dropped by more than 15%, attach a structured warning to the tool output:

```python
regression_warning = None
if post_gradient > 2.0 * max(pre_gradient, 0.01):
    regression_warning = {
        "type": "gradient_increased",
        "pre": pre_gradient,
        "post": post_gradient,
        "likely_cause": (
            "The background model may have over-fit the target signal. "
            "Common on extended nebulae and large galaxies."
        ),
        "suggested_actions": [
            "If method=graxpert: increase graxpert_options.smoothing (e.g. 0.9 or higher), "
            "or switch to method='polynomial' with degree=2.",
            "If method=polynomial: reduce polynomial_options.degree to 1, or accept "
            "that no gradient correction is needed and revert to the pre-gradient image.",
            "Revert to pre-gradient image if baseline gradient_magnitude was already < 0.05 "
            "(the image is essentially flat; further correction can only hurt).",
        ],
    }

if signal_coverage_pct_post < 0.85 * signal_coverage_pct_pre:
    regression_warning = {...target_signal_subtracted...}

summary = {
    "output_path": str(output_path),
    "pre_metrics": {"gradient_magnitude": pre_gradient, ...},
    "post_metrics": {"gradient_magnitude": post_gradient, ...},
    "regression_warning": regression_warning,
    ...
}
```

## Prompt-side lever

The NONLINEAR prompt (actually LINEAR, since gradient lives there) should include:

> If `remove_gradient` returns a `regression_warning`, do not proceed. Either
> re-run with different parameters (higher smoothing / lower polynomial degree)
> or switch `method`. A pre-gradient baseline `gradient_magnitude` below ~0.05
> means the data is already flat; in that case, the correct action is to skip
> gradient removal entirely.

This fixes the M20 failure mode at both layers: tool now offers alternatives, and prompt now forbids the "accept regression and move on" behavior.

## Effort

~70 lines of code in t09_gradient.py. Prompt update is ~8 lines in `prompts.py` LINEAR section.
