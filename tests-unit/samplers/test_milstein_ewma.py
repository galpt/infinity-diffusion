# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Galih Tama
"""Tests for the Milstein-EWMA sampler, its ComfyUI registration and its installer."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import comfy.k_diffusion.sampling as k_sampling
import comfy.model_sampling
import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler, get_ancestral_step, get_sigmas_karras, to_d

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "comfy-milstein-ewma.sh"
NODE_FILES = ["__init__.py", "milstein_ewma_diffusion.py"]

BASE_SIGMAS = torch.tensor([14.6146, 9.6825, 6.6780, 4.7746, 3.5221, 2.6666, 2.0606, 1.6156,
                            1.2768, 1.0097, 0.7913, 0.6055, 0.4397, 0.2780, 0.0292, 0.0])
HIRES_SIGMAS = torch.tensor([0.832, 0.739, 0.653, 0.571, 0.494, 0.418, 0.344, 0.267, 0.18, 0.029, 0.0])


def _load_module(name, path, **kwargs):
    """Import the file at path as module name.

    The module must be in sys.modules before it runs: dataclasses looks up
    the module of a class by name when the decorator processes the class.
    """
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mew = _load_module("milstein_ewma_diffusion_under_test", REPO_ROOT / "milstein_ewma_diffusion.py")


def _smooth(x):
    return torch.tanh(0.3 * x) + 0.05 * torch.sin(3.0 * x)


class FakeModel:
    """Stand-in for the wrapped ComfyUI model, with a call counter."""

    def __init__(self, model_sampling=None, output=_smooth):
        sampling = comfy.model_sampling.EPS() if model_sampling is None else model_sampling
        self.inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=sampling))
        self.output = output
        self.calls = 0
        self.outputs = []

    def __call__(self, x, sigma, **extra_args):
        self.calls += 1
        out = self.output(x)
        self.outputs.append(out)
        return out


class CountingNoise:
    """Seeded noise sampler that counts how often it is drawn from."""

    def __init__(self, x, seed):
        self._inner = default_noise_sampler(x, seed=seed)
        self.calls = 0

    def __call__(self, sigma, sigma_next):
        self.calls += 1
        return self._inner(sigma, sigma_next)


def _latent(sigmas, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((2, 4, 6, 6), generator=generator) * float(sigmas[0])


def _pattern(plan, steps):
    return "".join("F" if plan.uses_forecast(i) else "M" for i in range(steps))


def _forecast_steps(plan, steps):
    return [i for i in range(steps) if plan.uses_forecast(i)]


def _default_plan(sigmas):
    return mew.ForecastPlan(sigmas, mew.DEFAULT_FORECAST_RATIO, mew.DEFAULT_FORECAST_SIGMA)


def _reference_sample(model, x, sigmas, seed, alpha, k, forecast_steps=(), beta=0.8):
    """Euler ancestral steps with the diagonal Milstein noise, written from the formulas.

    On the steps in forecast_steps the model is skipped and the EWMA of its
    earlier outputs stands in for the denoised latent.
    """

    def m(y):
        return torch.clamp(1.0 + alpha * torch.tanh(y / k), min=1.0 - alpha, max=1.0 + alpha)

    h = 1e-3
    noise_sampler = default_noise_sampler(x, seed=seed)
    s_in = x.new_ones([x.shape[0]])
    level = None
    for i in range(len(sigmas) - 1):
        if i in forecast_steps:
            denoised = level
        else:
            denoised = model(x, sigmas[i] * s_in)
            level = denoised if level is None else beta * denoised + (1.0 - beta) * level
        if sigmas[i + 1] == 0:
            return denoised
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=1.0)
        x_next = x + to_d(x, sigmas[i], denoised) * (sigma_down - sigmas[i])
        up = float(sigma_up)
        dW = noise_sampler(sigmas[i], sigmas[i + 1]) * up
        slope = (m(x + h) - m(x - h)) / (2.0 * h)
        x_next = x_next + m(x) * dW
        x = x_next + 0.5 * m(x) * slope * (dW * dW - up ** 2)
    return x


def _make_comfyui(tmp_path):
    # The space checks that the installer quotes every path it touches.
    comfyui = tmp_path / "Comfy UI"
    (comfyui / "custom_nodes").mkdir(parents=True)
    return comfyui


def _run_installer(*args):
    return subprocess.run(["bash", str(INSTALLER), *map(str, args)], capture_output=True, text=True)


@pytest.fixture
def registry():
    """Load installed packs by name, then restore the global ComfyUI registration."""
    names = (comfy.samplers.KSAMPLER_NAMES, comfy.samplers.SAMPLER_NAMES)
    saved_names = [list(n) for n in names]
    missing = object()
    saved_attr = getattr(k_sampling, "sample_milstein_ewma", missing)
    loaded = []

    def load(name, node_dir):
        loaded.append(name)
        return _load_module(name, node_dir / "__init__.py", submodule_search_locations=[str(node_dir)])

    yield load

    for name in loaded:
        for key in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
            del sys.modules[key]
    for current, saved in zip(names, saved_names):
        current[:] = saved
    if saved_attr is missing:
        del k_sampling.sample_milstein_ewma
    else:
        k_sampling.sample_milstein_ewma = saved_attr


# Agreement with reference implementations

@pytest.mark.parametrize("alpha, k", [(0.15, 1.0), (0.5, 0.7)])
def test_zero_forecast_matches_reference_loop(alpha, k):
    """With forecasting off, the sampler reproduces a loop written from the formulas."""
    x = _latent(BASE_SIGMAS)
    got = mew.sample_milstein_ewma(FakeModel(), x.clone(), BASE_SIGMAS, {"seed": 7}, disable=True,
                                   alpha=alpha, k=k, forecast_ratio=0.0)
    want = _reference_sample(FakeModel(), x.clone(), BASE_SIGMAS, 7, alpha, k)
    assert torch.equal(got, want)


def test_default_run_matches_reference_loop_with_forecast():
    """With the defaults, the forecast steps feed the EWMA level into the same step formula."""
    x = _latent(BASE_SIGMAS)
    got = mew.sample_milstein_ewma(FakeModel(), x.clone(), BASE_SIGMAS, {"seed": 7}, disable=True)
    want = _reference_sample(FakeModel(), x.clone(), BASE_SIGMAS, 7, mew.DEFAULT_ALPHA, mew.DEFAULT_K,
                             forecast_steps={6, 7, 9, 10, 12, 13}, beta=mew.DEFAULT_BETA)
    assert torch.equal(got, want)


@pytest.mark.parametrize("sigmas", [BASE_SIGMAS, BASE_SIGMAS[:9], torch.tensor([4.0, 2.0, 2.0, 1.0, 0.0])],
                         ids=["full", "leftover-noise", "repeated-sigma"])
@pytest.mark.parametrize("eta", [1.0, 0.0])
def test_zero_alpha_zero_forecast_equals_euler_ancestral(sigmas, eta):
    """With alpha 0 and forecasting off, the result is sample_euler_ancestral bit for bit."""
    x = _latent(sigmas)
    got = mew.sample_milstein_ewma(FakeModel(), x.clone(), sigmas, {"seed": 7}, disable=True,
                                   eta=eta, alpha=0.0, forecast_ratio=0.0)
    want = k_sampling.sample_euler_ancestral(FakeModel(), x.clone(), sigmas, {"seed": 7},
                                             disable=True, eta=eta)
    assert torch.equal(got, want)


# Forecast plan

@pytest.mark.parametrize("sigmas, pattern", [
    (BASE_SIGMAS, "MMMMMMFFMFFMFFM"),
    (HIRES_SIGMAS, "MMFMFMFMFM"),
])
def test_default_plan_pattern_on_base_and_hires_schedules(sigmas, pattern):
    """The default plan gives the expected pattern on the 15-step base and 10-step hires schedules."""
    assert _pattern(_default_plan(sigmas), len(sigmas) - 1) == pattern


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.4, 0.6])
@pytest.mark.parametrize("steps", range(2, 41))
def test_plan_properties_on_karras_schedules(steps, ratio):
    """The plan keeps the ends, respects the sigma limit and the budget, and spreads its picks.

    Spreading means the picks end on the last candidate and the gaps
    between them differ by at most one, measured over the candidate run
    with a virtual pick just before it.
    """
    sigmas = get_sigmas_karras(steps, 0.0292, 14.6146)
    limit = mew.DEFAULT_FORECAST_SIGMA
    plan = mew.ForecastPlan(sigmas, ratio, limit)
    picked = _forecast_steps(plan, steps)
    candidates = [i for i in range(1, steps - 1) if float(sigmas[i]) < limit]
    budget = min(int(ratio * steps), len(candidates))

    assert 0 not in picked and steps - 1 not in picked
    assert all(float(sigmas[i]) < limit for i in picked)
    assert len(picked) == budget
    if any(b - a == 1 for a, b in zip(picked, picked[1:])):
        assert 2 * budget > len(candidates)
    if budget:
        assert picked[-1] == candidates[-1]
        gaps = [b - a for a, b in zip([candidates[0] - 1] + picked, picked)]
        assert max(gaps) - min(gaps) <= 1


def test_plan_without_candidates_forecasts_nothing():
    """When every sigma is above the limit, no step is forecast and every step calls the model."""
    plan = mew.ForecastPlan(BASE_SIGMAS, 0.4, 1e-6)
    assert _forecast_steps(plan, len(BASE_SIGMAS) - 1) == []


# Sampling loop

@pytest.mark.parametrize("sigmas, calls", [(BASE_SIGMAS, 9), (HIRES_SIGMAS, 6)])
def test_model_called_once_per_planned_step(sigmas, calls):
    """The model is called once for every step the default plan does not forecast."""
    steps = len(sigmas) - 1
    model = FakeModel()
    mew.sample_milstein_ewma(model, _latent(sigmas), sigmas, {"seed": 3}, disable=True)
    assert model.calls == steps - len(_forecast_steps(_default_plan(sigmas), steps)) == calls


@pytest.mark.parametrize("ratio", [0.0, mew.DEFAULT_FORECAST_RATIO])
def test_noise_drawn_once_per_nonterminal_step(ratio):
    """The noise sampler is drawn from on every step except the one that ends at zero."""
    x = _latent(BASE_SIGMAS)
    noise = CountingNoise(x, seed=5)
    mew.sample_milstein_ewma(FakeModel(), x, BASE_SIGMAS, {}, disable=True, noise_sampler=noise,
                             forecast_ratio=ratio)
    assert noise.calls == len(BASE_SIGMAS) - 2


def test_callback_denoised_is_ewma_on_forecast_steps():
    """Forecast steps report the EWMA of the model outputs so far, model steps report the output."""
    sigmas = BASE_SIGMAS
    model = FakeModel()
    seen = []
    mew.sample_milstein_ewma(model, _latent(sigmas), sigmas, {"seed": 3}, disable=True,
                             callback=lambda info: seen.append((info["i"], info["denoised"])))
    plan = _default_plan(sigmas)
    beta = mew.DEFAULT_BETA
    level = None
    outputs = iter(model.outputs)
    assert [i for i, _ in seen] == list(range(len(sigmas) - 1))
    for i, denoised in seen:
        if plan.uses_forecast(i):
            assert torch.allclose(denoised, level, rtol=0.0, atol=1e-6)
        else:
            out = next(outputs)
            assert torch.equal(denoised, out)
            level = out if level is None else beta * out + (1.0 - beta) * level
    assert next(outputs, None) is None


def test_schedule_ending_above_zero_runs_to_a_noisy_latent():
    """A schedule that stops above zero still calls the model on its last step and stays finite."""
    sigmas = BASE_SIGMAS[:9]
    model = FakeModel()
    seen = []
    result = mew.sample_milstein_ewma(model, _latent(sigmas), sigmas, {"seed": 3}, disable=True,
                                      callback=lambda info: seen.append(info["denoised"]))
    last = len(sigmas) - 2
    assert not _default_plan(sigmas).uses_forecast(last)
    assert torch.equal(seen[-1], model.outputs[-1])
    assert torch.isfinite(result).all()
    assert not torch.equal(result, seen[-1])


def test_input_latent_is_not_modified():
    """The sampler leaves the latent it was given untouched."""
    x = _latent(BASE_SIGMAS)
    before = x.clone()
    mew.sample_milstein_ewma(FakeModel(), x, BASE_SIGMAS, {"seed": 3}, disable=True)
    assert torch.equal(x, before)


# EWMA forecast

def test_ewma_forecast_follows_closed_form():
    """observe and predict give the closed-form EWMA, and predict fails before any observation."""
    beta = 0.8
    a, b, c = torch.tensor([1.0, -2.0]), torch.tensor([3.0, 0.5]), torch.tensor([-1.0, 4.0])
    forecast = mew.EwmaForecast(beta)
    with pytest.raises(RuntimeError):
        forecast.predict()

    forecast.observe(a)
    assert torch.equal(forecast.predict(), a)
    forecast.observe(b)
    assert torch.allclose(forecast.predict(), beta * b + (1 - beta) * a)
    forecast.observe(c)
    assert torch.allclose(forecast.predict(), beta * c + beta * (1 - beta) * b + (1 - beta) ** 2 * a)


# Diagonal Milstein noise

def test_zero_alpha_noise_is_unit_scale_and_flat():
    """With alpha 0 the scale is exactly one and the slope exactly zero."""
    x = torch.linspace(-5.0, 5.0, 101)
    noise = mew.DiagonalMilsteinNoise(alpha=0.0, k=1.0)
    assert torch.equal(noise.scale(x), torch.ones_like(x))
    assert torch.equal(noise.slope(x), torch.zeros_like(x))


@pytest.mark.parametrize("alpha, k", [(0.15, 1.0), (0.6, 0.5)])
def test_slope_matches_analytic_derivative(alpha, k):
    """The central difference agrees with (alpha / k) * (1 - tanh(x / k)^2) to 1e-3."""
    x = torch.linspace(-6.0, 6.0, 241)
    want = (alpha / k) * (1.0 - torch.tanh(x / k) ** 2)
    got = mew.DiagonalMilsteinNoise(alpha, k).slope(x)
    assert (got - want).abs().max() < 1e-3


def test_milstein_correction_has_zero_mean():
    """The correction averages to zero over dW ~ N(0, var), within five standard errors.

    At a fixed x the correction is c * (dW^2 - var) with c = 0.5 * m * m' and
    Var[dW^2 - var] = 2 * var^2, so the mean of n draws has a standard error
    of c * var * sqrt(2 / n). A missing variance term would shift the mean by
    c * var, which is far outside the tolerance.
    """
    alpha, k, x0, var, n = 0.5, 1.0, 0.5, 1.0, 200_000
    noise = mew.DiagonalMilsteinNoise(alpha, k)
    generator = torch.Generator().manual_seed(11)
    x = torch.full((n,), x0)
    dW = torch.randn(n, generator=generator) * math.sqrt(var)

    correction = noise.perturb(torch.zeros(n), x, dW, var) - noise.scale(x) * dW

    m = 1.0 + alpha * math.tanh(x0 / k)
    slope = (alpha / k) * (1.0 - math.tanh(x0 / k) ** 2)
    tolerance = 5.0 * 0.5 * m * slope * var * math.sqrt(2.0 / n)
    assert abs(float(correction.double().mean())) < tolerance


# Rejected inputs

@pytest.mark.parametrize("name, value", [
    ("alpha", -0.1), ("alpha", 1.0), ("alpha", float("nan")),
    ("k", 0.0), ("k", -1.0), ("k", float("inf")),
    ("beta", 0.0), ("beta", 1.5), ("beta", float("nan")),
    ("forecast_ratio", -0.1), ("forecast_ratio", 1.0),
    ("forecast_sigma", 0.0), ("forecast_sigma", -1.0),
    ("eta", -1.0), ("eta", float("nan")),
    ("s_noise", -1.0),
])
def test_invalid_parameter_raises_value_error(name, value):
    """Each out-of-range sampler parameter is rejected before any model call."""
    model = FakeModel()
    with pytest.raises(ValueError):
        mew.sample_milstein_ewma(model, _latent(BASE_SIGMAS), BASE_SIGMAS, {}, disable=True,
                                 **{name: value})
    assert model.calls == 0


@pytest.mark.parametrize("sigmas", [
    torch.tensor([[2.0, 1.0, 0.0]]),
    torch.tensor([1.0]),
    torch.tensor([2.0, float("nan"), 0.0]),
    torch.tensor([1.0, 2.0, 0.0]),
    torch.tensor([2.0, 0.0, 0.0]),
    torch.tensor([2.0, 1.0, -0.5]),
], ids=["2d", "single", "nan", "increasing", "zero-before-end", "negative-last"])
def test_invalid_sigmas_raise_value_error(sigmas):
    """A schedule that is not 1-D, finite, non-increasing and positive before its end is rejected."""
    model = FakeModel()
    with pytest.raises(ValueError):
        mew.sample_milstein_ewma(model, torch.zeros(1, 4, 2, 2), sigmas, {}, disable=True)
    assert model.calls == 0


def test_nan_model_output_raises_runtime_error():
    """A model that returns NaN stops the run with an error that names the model output."""
    model = FakeModel(output=lambda x: torch.full_like(x, float("nan")))
    with pytest.raises(RuntimeError, match="model output"):
        mew.sample_milstein_ewma(model, _latent(BASE_SIGMAS), BASE_SIGMAS, {}, disable=True)


def test_const_model_is_rejected():
    """A rectified-flow (CONST) model raises ValueError before any model call."""
    model = FakeModel(model_sampling=comfy.model_sampling.CONST())
    with pytest.raises(ValueError):
        mew.sample_milstein_ewma(model, _latent(BASE_SIGMAS), BASE_SIGMAS, {}, disable=True)
    assert model.calls == 0


# Registration and installer

def test_installer_installs_and_uninstalls(tmp_path):
    """Install copies the two package files, repeats cleanly, and uninstall removes them."""
    comfyui = _make_comfyui(tmp_path)
    node_dir = comfyui / "custom_nodes" / "milstein-ewma-diffusion"

    assert _run_installer(comfyui).returncode == 0
    assert sorted(p.name for p in node_dir.iterdir()) == NODE_FILES
    assert (node_dir / "__init__.py").read_bytes() == (REPO_ROOT / "custom_node" / "__init__.py").read_bytes()
    assert (node_dir / NODE_FILES[1]).read_bytes() == (REPO_ROOT / NODE_FILES[1]).read_bytes()

    assert _run_installer(comfyui, "install").returncode == 0
    assert sorted(p.name for p in node_dir.iterdir()) == NODE_FILES

    assert _run_installer(comfyui, "uninstall").returncode == 0
    assert not node_dir.exists()
    assert _run_installer(comfyui, "uninstall").returncode == 0


def test_installer_rejects_bad_arguments(tmp_path):
    """A missing path, a tree without custom_nodes and an unknown mode all exit non-zero."""
    comfyui = _make_comfyui(tmp_path)
    bare = tmp_path / "bare"
    bare.mkdir()

    assert _run_installer().returncode != 0
    assert _run_installer(bare).returncode != 0
    assert _run_installer(comfyui, "bogus").returncode != 0
    assert not (comfyui / "custom_nodes" / "milstein-ewma-diffusion").exists()


def test_installed_package_registers_sampler_once(tmp_path, registry):
    """Importing the installed package twice registers milstein_ewma once and adds no nodes."""
    comfyui = _make_comfyui(tmp_path)
    assert _run_installer(comfyui).returncode == 0
    node_dir = comfyui / "custom_nodes" / "milstein-ewma-diffusion"

    first = registry("installed_pack_first", node_dir)
    second = registry("installed_pack_second", node_dir)

    assert k_sampling.sample_milstein_ewma is second.sample_milstein_ewma
    assert comfy.samplers.KSAMPLER_NAMES.count("milstein_ewma") == 1
    assert comfy.samplers.SAMPLER_NAMES.count("milstein_ewma") == 1
    assert first.NODE_CLASS_MAPPINGS == {} and second.NODE_CLASS_MAPPINGS == {}
