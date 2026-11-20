# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Galih Tama
"""Milstein-EWMA sampler for ComfyUI.

Milstein-EWMA integrates the reverse diffusion as an ancestral SDE, the
same way sample_euler_ancestral does: each step follows the Euler drift of
the probability-flow ODE down to sigma_down and then re-injects noise up to
the next sigma. It adds two things to that scheme.

Diffusion term
--------------
The re-injected noise is scaled by a bounded multiplier that depends on the
current state,

    m(x) = clip(1 + alpha * tanh(x / k), 1 - alpha, 1 + alpha),

and corrected with the elementwise Milstein term

    0.5 * m(x) * m'(x) * (dW^2 - Var[dW]).

The correction has zero mean, so it reshapes the step noise without biasing
it. Each latent element is treated as its own scalar SDE, which is why the
cross terms of the full Milstein scheme do not appear. m'(x) is a central
difference of m and costs no model evaluation. With alpha = 0, m is one,
the correction vanishes and every step is an Euler ancestral step.

Forecast of the denoised output
-------------------------------
Nearly all of the sampling time is spent in the model, so making the step
arithmetic cheaper cannot speed the sampler up in any noticeable way. What
can is calling the model less often. On a share of the steps the sampler
skips the model and uses a forecast of its denoised output instead: an
exponentially weighted moving average (EWMA) of the outputs the model has
returned so far,

    level = beta * denoised + (1 - beta) * level.

A trend term looks like the natural way to see ahead, but with eta = 1 the
fresh noise of every step makes the denoised output jump by a large
fraction of its own size from one step to the next. An extrapolated trend
overshoots more often than it helps, while a smoothed level stays close.

Where the forecast is used matters more than how it is made. While sigma is
large the model is still settling the composition, and a forecast there
changes the final image. Once sigma has dropped to a few units the
structure is in place, and a forecast mostly changes small details. The
plan therefore picks int(forecast_ratio * steps)
steps among those whose sigma is below forecast_sigma and spreads them as
evenly as the count allows. Two forecast steps are adjacent only when more
than half of those steps are picked.

The first step always calls the model because the EWMA needs a value to
start from. The last step calls it too: on a full run its denoised output
is the result, and on a run that stops early it sets the leftover noise
the next sampler continues from. The plan depends on the sigma schedule
alone, so the number of model calls is known before sampling starts.

Rectified-flow (CONST) models are rejected. Their ancestral step splits the
noise differently and this sampler does not implement that split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from comfy.k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d, trange
from comfy.model_sampling import CONST

DEFAULT_ALPHA = 0.15
DEFAULT_K = 1.0
DEFAULT_BETA = 0.8
DEFAULT_FORECAST_RATIO = 0.4
DEFAULT_FORECAST_SIGMA = 3.0

# Half-width of the central difference that estimates m'(x).
SLOPE_STEP = 1e-3


def _finite(name: str, value) -> float:
    """Return value as a float, raising when it is NaN or infinite."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise when tensor holds NaN or infinity.

    A NaN from the model, usually an fp16 overflow, would otherwise run
    through every later step and come out as a black image.
    """
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{name} is not finite")


def _check_sigmas(sigmas: torch.Tensor) -> None:
    """Raise unless sigmas is a schedule this sampler can step through.

    Every sigma before the last must be positive because the drift divides
    by it. A repeated sigma is harmless: the step between the two copies
    has neither drift nor noise. The last sigma is zero for a full run and
    positive when KSamplerAdvanced stops early with leftover noise, so it
    only has to be non-negative.
    """
    if sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("sigmas must be a 1-D schedule with at least two entries")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("sigmas must be finite")
    if not bool((sigmas[1:] <= sigmas[:-1]).all()):
        raise ValueError("sigmas must not increase")
    if not bool((sigmas[:-1] > 0.0).all()) or float(sigmas[-1]) < 0.0:
        raise ValueError("sigmas must be positive before the last entry and non-negative at the end")


@dataclass(frozen=True)
class DiagonalMilsteinNoise:
    """State-scaled step noise and its diagonal Milstein correction.

    m and m' are evaluated in float32 whatever the dtype of the latent. The
    central difference step is smaller than the spacing of float16 values
    from |x| = 2 upwards, so in half precision x + SLOPE_STEP would round
    back to x and the slope would come out as zero.
    """

    alpha: float
    k: float

    def __post_init__(self):
        if not 0.0 <= _finite("alpha", self.alpha) < 1.0:
            raise ValueError("alpha must lie in [0, 1)")
        if not _finite("k", self.k) > 0.0:
            raise ValueError("k must be positive")

    def _m(self, xf: torch.Tensor) -> torch.Tensor:
        """Return m at a float32 tensor.

        tanh keeps m inside [1 - alpha, 1 + alpha] already. The clamp only
        stops rounding from stepping past the bounds.
        """
        raw = 1.0 + self.alpha * torch.tanh(xf / self.k)
        return torch.clamp(raw, min=1.0 - self.alpha, max=1.0 + self.alpha)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Return m(x)."""
        return self._m(x.to(torch.float32)).to(dtype=x.dtype)

    def slope(self, x: torch.Tensor) -> torch.Tensor:
        """Return m'(x) as a central difference of m."""
        xf = x.to(torch.float32)
        return ((self._m(xf + SLOPE_STEP) - self._m(xf - SLOPE_STEP)) / (2.0 * SLOPE_STEP)).to(dtype=x.dtype)

    def perturb(self, x_next: torch.Tensor, x: torch.Tensor, dW: torch.Tensor, variance: float) -> torch.Tensor:
        """Add m(x) * dW and the Milstein correction to x_next.

        m and m' are taken at x, the state at the start of the step, as the
        scheme requires. variance is Var[dW], which makes the correction
        zero-mean.
        """
        m = self.scale(x)
        x_next = x_next + m * dW
        return x_next + 0.5 * m * self.slope(x) * (dW * dW - variance)


class ForecastPlan:
    """The steps of a schedule that use the forecast instead of the model.

    Candidates are the steps strictly between the first and the last whose
    sigma is below sigma_limit. Because sigmas do not increase, they form
    one run at the end of the schedule. Out of n candidates a budget of
    b = min(int(ratio * steps), n) is picked, candidate j being picked when
    floor((j + 1) * b / n) > floor(j * b / n). This is the rounding
    Bresenham's line algorithm uses: the gaps between picks differ by at
    most one, the last candidate is always picked when b > 0, and two picks
    are adjacent only when b > n / 2.
    """

    def __init__(self, sigmas: torch.Tensor, ratio: float, sigma_limit: float):
        values = sigmas.tolist()
        steps = len(values) - 1
        candidates = [i for i in range(1, steps - 1) if values[i] < sigma_limit]
        n = len(candidates)
        budget = min(int(ratio * steps), n)
        self._forecast_steps = frozenset(
            step for j, step in enumerate(candidates) if (j + 1) * budget // n > j * budget // n
        )

    def uses_forecast(self, step: int) -> bool:
        """Return True when step takes its denoised output from the forecast."""
        return step in self._forecast_steps


class EwmaForecast:
    """EWMA of the denoised outputs returned by the model."""

    def __init__(self, beta: float):
        self._beta = beta
        self._level = None

    def observe(self, denoised: torch.Tensor) -> None:
        """Fold a model output into the average."""
        if self._level is None:
            self._level = denoised
        else:
            self._level = self._beta * denoised + (1.0 - self._beta) * self._level

    def predict(self) -> torch.Tensor:
        """Return the current level as the forecast of the next output."""
        if self._level is None:
            raise RuntimeError("forecast requested before any model output was observed")
        return self._level


@dataclass(frozen=True)
class MilsteinEwmaSampler:
    """Ancestral diagonal Milstein sampler with an EWMA forecast."""

    diffusion: DiagonalMilsteinNoise
    eta: float
    s_noise: float
    beta: float
    forecast_ratio: float
    forecast_sigma: float

    def __post_init__(self):
        if not _finite("eta", self.eta) >= 0.0:
            raise ValueError("eta must be non-negative")
        if not _finite("s_noise", self.s_noise) >= 0.0:
            raise ValueError("s_noise must be non-negative")
        if not 0.0 < _finite("beta", self.beta) <= 1.0:
            raise ValueError("beta must lie in (0, 1]")
        if not 0.0 <= _finite("forecast_ratio", self.forecast_ratio) < 1.0:
            raise ValueError("forecast_ratio must lie in [0, 1)")
        if not _finite("forecast_sigma", self.forecast_sigma) > 0.0:
            raise ValueError("forecast_sigma must be positive")

    def sample(self, model, x: torch.Tensor, sigmas: torch.Tensor, extra_args: dict,
               callback=None, disable=None, noise_sampler=None) -> torch.Tensor:
        """Step x through sigmas and return the final latent.

        model is called as model(x, sigma, **extra_args) and returns the
        denoised latent. noise_sampler defaults to the seeded ComfyUI
        sampler.
        """
        _check_sigmas(sigmas)
        _check_finite("initial latent", x)
        if noise_sampler is None:
            noise_sampler = default_noise_sampler(x, seed=extra_args.get("seed"))
        plan = ForecastPlan(sigmas, self.forecast_ratio, self.forecast_sigma)
        forecast = EwmaForecast(self.beta)
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            if plan.uses_forecast(i):
                denoised = forecast.predict()
            else:
                denoised = model(x, sigma * s_in, **extra_args)
                _check_finite("model output", denoised)
                forecast.observe(denoised)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            x = self._step(x, sigma, sigma_next, denoised, noise_sampler)
            _check_finite("sampler state", x)
        return x

    def _step(self, x: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor,
              denoised: torch.Tensor, noise_sampler) -> torch.Tensor:
        """Advance x from sigma to sigma_next.

        A step that ends at zero noise returns the denoised latent. Every
        other step draws from the noise sampler, even when it adds no noise,
        as sample_euler_ancestral does. A seed then gives the same noise
        sequence whatever the plan, eta or repeated sigmas in the schedule.
        """
        if sigma_next == 0:
            return denoised
        sigma_down, sigma_up = get_ancestral_step(sigma, sigma_next, eta=self.eta)
        x_next = x + to_d(x, sigma, denoised) * (sigma_down - sigma)
        noise = noise_sampler(sigma, sigma_next)
        if sigma_up == 0:
            return x_next
        std = self.s_noise * float(sigma_up)
        return self.diffusion.perturb(x_next, x, noise * std, std ** 2)


@torch.no_grad()
def sample_milstein_ewma(model, x, sigmas, extra_args=None, callback=None, disable=None,
                         eta=1.0, s_noise=1.0, noise_sampler=None,
                         alpha=DEFAULT_ALPHA, k=DEFAULT_K, beta=DEFAULT_BETA,
                         forecast_ratio=DEFAULT_FORECAST_RATIO, forecast_sigma=DEFAULT_FORECAST_SIGMA):
    """Sample with Milstein-EWMA. This is the function KSampler calls.

    The signature follows the k-diffusion samplers in ComfyUI. The model
    type is read the way sample_euler_ancestral reads it, so the check
    holds for every caller that can reach a k-diffusion sampler.
    """
    if isinstance(model.inner_model.inner_model.model_sampling, CONST):
        raise ValueError("milstein_ewma does not support rectified-flow (CONST) models")
    sampler = MilsteinEwmaSampler(
        diffusion=DiagonalMilsteinNoise(alpha=alpha, k=k),
        eta=eta,
        s_noise=s_noise,
        beta=beta,
        forecast_ratio=forecast_ratio,
        forecast_sigma=forecast_sigma,
    )
    return sampler.sample(model, x, sigmas, {} if extra_args is None else extra_args,
                          callback, disable, noise_sampler)
