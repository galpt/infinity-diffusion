"""Align Your Steps schedule for SDXL with log linear interpolation.

The paper optimizes timesteps by minimizing an upper bound on the KL
divergence between the true data path and the discretized sampling path.
The SDXL table below is the published ten step result for that objective.
This module keeps that table frozen and stretches it to any step count
with log linear interpolation in sigma space over the index domain.
Only steps selects the count, so there are no extra knobs to tune.

The schedule stays strictly decreasing with an exact terminal zero and
length steps plus one. A single step uses the first table entry with zero.
The ten step schedule uses the first ten entries with zero, so the final
table entry is dropped to keep the length invariant and to avoid a near
zero duplicate before the terminal zero. Longer and shorter schedules
interpolate the first ten entries and then append zero. The table itself
is never mutated and repeated calls give identical results.
"""

from __future__ import annotations

import math

import torch


__all__ = [
    "SDXL_TABLE",
    "SDXL_SIGMA_MAX",
    "loglinear_interp",
    "nv_ays_sigmas_for_steps",
]
__version__ = "1.0.0"


# Published SDXL ten step result with an extra trailing knot. The first ten
# entries drive all schedules. The final entry is kept for reference only.
SDXL_TABLE = (
    14.615,
    6.315,
    3.771,
    2.181,
    1.342,
    0.862,
    0.555,
    0.380,
    0.234,
    0.113,
    0.029,
)


# First table entry which starts every schedule.
SDXL_SIGMA_MAX = 14.615


def loglinear_interp(knots, n):
    """Interpolate positive knots in log space over the index domain.

    Knots are copied and never mutated. The index domain runs from zero to
    the last knot index. When the target count equals the knot count the
    result equals the input. When the target count is smaller the result
    is a uniform index grid mapped through log space. When the target
    count is larger the result keeps every input knot and fills the gaps
    with even log splits, so the input stays a subset of the output.
    The output is strictly decreasing when the input is strictly
    decreasing. All values stay positive and deterministic.
    """
    count = int(n)
    if count < 1:
        raise ValueError(f"count must be >=1, got {count}")
    vals = [float(v) for v in list(knots)]
    if len(vals) < 2:
        raise ValueError("knots must hold at least 2 values")
    for v in vals:
        if not v > 0.0:
            raise ValueError("knots must be positive")
    if not all(a > b for a, b in zip(vals, vals[1:])):
        raise ValueError("knots must be strictly decreasing")
    if count == len(vals):
        return [float(v) for v in vals]
    logs = [math.log(float(v)) for v in vals]
    size = len(vals)
    if count < size:
        out = []
        for i in range(count):
            if count == 1:
                pos = 0.0
            else:
                pos = float(i) * float(size - 1) / float(count - 1)
            low = int(math.floor(pos))
            high = int(math.ceil(pos))
            if low == high:
                out.append(math.exp(logs[low]))
            else:
                frac = pos - float(low)
                out.append(math.exp(logs[low] * (1.0 - frac) + logs[high] * frac))
        return [float(v) for v in out]
    extras = int(count - size)
    gaps = [float(logs[i] - logs[i + 1]) for i in range(size - 1)]
    base = int(extras // (size - 1))
    rem = int(extras % (size - 1))
    order = sorted(range(size - 1), key=lambda i: (-gaps[i], i))
    extra_per_gap = [int(base) for _ in range(size - 1)]
    for k in range(rem):
        extra_per_gap[order[k]] += 1
    out = [float(vals[0])]
    for i in range(size - 1):
        parts = int(extra_per_gap[i] + 1)
        for j in range(1, parts + 1):
            frac = float(j) / float(parts)
            out.append(math.exp(logs[i] * (1.0 - frac) + logs[i + 1] * frac))
    return [float(v) for v in out]


def nv_ays_sigmas_for_steps(steps):
    """Build AYS sigmas for the requested step count.

    Only steps selects the count. The result holds steps non zero sigmas
    plus a terminal zero, so length is steps plus one. Values are strictly
    decreasing with an exact terminal zero. A single step uses the first
    table entry with zero. Ten steps use the first ten entries with zero.
    The trailing table entry is dropped there to keep the length invariant
    and to avoid a near zero duplicate before the terminal zero. Any other
    count interpolates the first ten entries in log space and appends zero.
    The table is never mutated and repeated calls match exactly.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >=1, got {steps}")
    if steps == 1:
        return torch.tensor([float(SDXL_TABLE[0]), 0.0], dtype=torch.float32)
    if steps == 10:
        head = [float(v) for v in list(SDXL_TABLE[:10])]
        head.append(0.0)
        return torch.tensor(head, dtype=torch.float32)
    head_knots = [float(v) for v in list(SDXL_TABLE[:10])]
    interp = loglinear_interp(head_knots, steps)
    if len(interp) != steps:
        raise ValueError("interpolation failed to match steps")
    interp.append(0.0)
    sigmas = torch.tensor(interp, dtype=torch.float32)
    if len(sigmas) != steps + 1:
        raise ValueError("schedule length must be steps plus one")
    if float(sigmas[-1].item()) != 0.0:
        raise ValueError("terminal sigma must be zero")
    if not bool(torch.all(sigmas[:-1] > sigmas[1:])):
        raise ValueError("sigmas must be strictly decreasing")
    return sigmas
