"""Tests for seniourious-pure stage scheduler. Synthetic data only."""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import types

import torch


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CORE_FILE = _ROOT / "seniourious_pure_diffusion.py"
_NODE_FILE = _ROOT / "custom_node" / "__init__.py"


def _load_core():
    spec = importlib.util.spec_from_file_location("seniourious_pure_diffusion", str(_CORE_FILE))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load_core()


class _FakeSampling:
    """Minimal model sampling with invertible log mapping for tests."""

    def __init__(self) -> None:
        self.sigma_min = torch.tensor(0.0291675)
        self.sigma_max = torch.tensor(14.614642)

    def timestep(self, s: torch.Tensor) -> torch.Tensor:
        import math

        value = float(s.view(-1)[0].item()) if isinstance(s, torch.Tensor) else float(s)
        low = float(self.sigma_min.item())
        high = float(self.sigma_max.item())
        if value <= 0.0:
            return torch.tensor([0.0])
        position = 1000.0 * (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
        return torch.tensor([position])

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        import math

        value = float(t.view(-1)[0].item()) if isinstance(t, torch.Tensor) else float(t)
        low = float(self.sigma_min.item())
        high = float(self.sigma_max.item())
        log_sigma = math.log(low) + (value / 1000.0) * (math.log(high) - math.log(low))
        return torch.tensor([math.exp(log_sigma)])


def _fake_model(x: torch.Tensor, sigma, **kwargs) -> torch.Tensor:
    return 0.5 * x


def test_registration() -> None:
    """Scheduler and sampler names are registered without extra logic."""
    mock_samplers = types.ModuleType("comfy.samplers")
    mock_samplers.SCHEDULER_HANDLERS = {}
    mock_samplers.SCHEDULER_NAMES = []
    mock_samplers.KSAMPLER_NAMES = ["euler", "dpmpp_2m"]
    mock_samplers.SAMPLER_NAMES = ["euler", "dpmpp_2m"]

    class _Handler:
        def __init__(self, fn) -> None:
            self.handler = fn

    mock_samplers.SchedulerHandler = _Handler
    mock_sampling = types.ModuleType("comfy.k_diffusion.sampling")
    for name in ("sample_euler", "sample_dpm_2_ancestral", "sample_dpmpp_2m"):
        setattr(mock_sampling, name, lambda *a, **k: None)
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.samplers"] = mock_samplers
    sys.modules["comfy.k_diffusion"] = types.ModuleType("comfy.k_diffusion")
    sys.modules["comfy.k_diffusion.sampling"] = mock_sampling
    try:
        spec = importlib.util.spec_from_file_location("seniourious_pure_node", str(_NODE_FILE))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key in ("comfy", "comfy.samplers", "comfy.k_diffusion", "comfy.k_diffusion.sampling"):
            sys.modules.pop(key, None)
    assert "seniourious-pure" in mock_samplers.SCHEDULER_HANDLERS
    assert "seniourious-pure" in mock_samplers.SCHEDULER_NAMES
    assert "seniourious-pure" in mock_samplers.SAMPLER_NAMES
    assert "seniourious-pure" in mock_samplers.KSAMPLER_NAMES
    assert getattr(mock_sampling, "sample_seniourious-pure") is not None
    assert "seniourious" not in mock_samplers.SCHEDULER_NAMES
    assert "seniourious" not in mock_samplers.SAMPLER_NAMES


def test_invariants_steps() -> None:
    """Length, terminal zero, and strict decrease hold for key counts."""
    sampling = _FakeSampling()
    for steps in (1, 4, 20, 50):
        sigmas = S.seniourious_pure_scheduler(sampling, steps)
        assert len(sigmas) == steps + 1
        assert float(sigmas[-1].item()) == 0.0
        assert bool(torch.all(sigmas[:-1] > sigmas[1:]))


def test_nfe_table() -> None:
    """Per step counts match the solver cost without extra cost."""
    assert S.nfe_per_step("euler") == 1
    assert S.nfe_per_step("dpm_2_ancestral") == 2
    assert S.nfe_per_step("dpmpp_2m") == 1
    assert S.count_nfe(1) == 1
    assert S.count_nfe(4) == 5
    assert S.count_nfe(20) == 26
    assert S.count_nfe(50) == 66


def test_split_and_order() -> None:
    """Split is one third early with stochastic early and late close."""
    assert S.STAGE_TABLE[0]["sampler"] == "dpm_2_ancestral"
    assert S.STAGE_TABLE[0]["kind"] == "sde"
    assert S.STAGE_TABLE[1]["sampler"] == "dpmpp_2m"
    assert S.STAGE_TABLE[1]["kind"] == "ode"
    for steps in (1, 4, 20, 50):
        sde_steps, ode_steps = S.split_steps(steps)
        assert sde_steps + ode_steps == steps
        assert ode_steps >= 1
        if steps == 1:
            assert (sde_steps, ode_steps) == (0, 1)
        else:
            assert sde_steps == steps // 3


def test_knob_free_signatures() -> None:
    """Public entry points expose only steps with no extra knobs."""
    sig = inspect.signature(S.seniourious_pure_scheduler)
    assert list(sig.parameters.keys()) == ["model_sampling", "steps"]
    sig = inspect.signature(S.sample_seniourious_pure)
    assert list(sig.parameters.keys()) == ["model", "x", "sigmas", "extra_args", "callback", "disable"]
    sig = inspect.signature(S.split_steps)
    assert list(sig.parameters.keys()) == ["steps"]


def test_delegation_slices() -> None:
    """Stages call built-in solvers with one shared node."""
    original = S._sampler_fn

    def _run(steps: int) -> list[tuple[str, int, torch.Tensor]]:
        calls: list[tuple[str, int, torch.Tensor]] = []

        def _mock(name: str):
            def _fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
                calls.append((name, len(sigmas), sigmas.clone()))
                local = x
                s_in = local.new_ones([local.shape[0]])
                for i in range(len(sigmas) - 1):
                    denoised = model(local, sigmas[i] * s_in, **(extra_args or {}))
                    if callback is not None:
                        callback({"x": local, "i": i, "sigma": sigmas[i], "denoised": denoised})
                    if float(sigmas[i + 1].item()) == 0.0:
                        local = denoised
                    else:
                        direction = (local - denoised) / sigmas[i]
                        local = local + direction * (sigmas[i + 1] - sigmas[i])
                return local

            return _fn

        try:
            S._sampler_fn = _mock  # type: ignore
            sigmas = S.seniourious_pure_scheduler(_FakeSampling(), steps)
            start = torch.ones(1, 2, 4, 4)
            S.sample_seniourious_pure(_fake_model, start.clone(), sigmas.clone(), disable=True)
        finally:
            S._sampler_fn = original  # type: ignore
        return calls

    for steps in (4, 6, 20):
        calls = _run(steps)
        if steps == 1:
            continue
        assert calls[0][0] == str(S.STAGE_TABLE[0]["sampler"])
        assert calls[1][0] == str(S.STAGE_TABLE[1]["sampler"])
        sde_steps, ode_steps = S.split_steps(steps)
        if sde_steps == 0:
            assert len(calls) == 1
            continue
        assert calls[0][1] == sde_steps + 1
        assert calls[1][1] == ode_steps + 1
        assert calls[0][1] + calls[1][1] == steps + 2
        assert float(calls[0][2][-1].item()) == float(calls[1][2][0].item())


def test_nfe_mock_counts() -> None:
    """Mock evaluations match the fixed schedule total."""
    counts = {"early": 0, "late": 0}
    original = S._sampler_fn

    def _early(model, x, sigmas, extra_args=None, callback=None, disable=None):
        for _ in range(len(sigmas) - 1):
            for _ in range(2):
                counts["early"] += 1
                model(x, sigmas[0:1], **(extra_args or {}))
        return x

    def _late(model, x, sigmas, extra_args=None, callback=None, disable=None):
        for _ in range(len(sigmas) - 1):
            counts["late"] += 1
            model(x, sigmas[0:1], **(extra_args or {}))
        return x

    def _mock(name: str):
        return _early if name == str(S.STAGE_TABLE[0]["sampler"]) else _late

    try:
        S._sampler_fn = _mock  # type: ignore
        sigmas = torch.tensor([5.0, 3.0, 1.5, 0.7, 0.0])
        S.sample_seniourious_pure(_fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), disable=True)
    finally:
        S._sampler_fn = original  # type: ignore
    assert counts["early"] + counts["late"] == S.count_nfe(4)


def test_gap_uniformity() -> None:
    """Timestep gaps stay even from uniform spacing."""
    sampling = _FakeSampling()
    for steps in (20, 50):
        sigmas = S.seniourious_pure_scheduler(sampling, steps)
        times = [float(sampling.timestep(s).item()) for s in sigmas[:-1]]
        gaps = [abs(a - b) for a, b in zip(times, times[1:])]
        assert len(gaps) == steps - 1
        assert min(gaps) > 0.0
        assert max(gaps) / min(gaps) <= 2.0 + 1e-6


def test_extra_args_shared() -> None:
    """Both stages receive the same extra_args object."""
    seen: list[object] = []
    original = S._sampler_fn

    def _mock(name: str):
        def _fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            seen.append(extra_args)
            return x

        return _fn

    try:
        S._sampler_fn = _mock  # type: ignore
        sigmas = S.seniourious_pure_scheduler(_FakeSampling(), 6)
        given: dict = {"cfg": 1.0}
        S.sample_seniourious_pure(
            _fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), extra_args=given, disable=True
        )
    finally:
        S._sampler_fn = original  # type: ignore
    assert len(seen) == 2
    assert seen[0] is given
    assert seen[1] is given


def test_callback_global_index() -> None:
    """Late callbacks continue the global step count."""
    original = S._sampler_fn

    def _mock(name: str):
        def _fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            for i in range(len(sigmas) - 1):
                if callback is not None:
                    callback({"x": x, "i": i, "sigma": sigmas[i]})
            return x

        return _fn

    try:
        S._sampler_fn = _mock  # type: ignore
        for steps in (4, 6, 20):
            seen: list[int] = []

            def _cb(data, _seen=seen) -> None:
                _seen.append(int(data["i"]))

            sigmas = S.seniourious_pure_scheduler(_FakeSampling(), steps)
            seen.clear()
            S.sample_seniourious_pure(
                _fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), callback=_cb, disable=True
            )
            assert seen == list(range(steps))
    finally:
        S._sampler_fn = original  # type: ignore


def test_single_step_euler() -> None:
    """A single step calls Euler directly."""
    seen: list[str] = []
    original = S._sampler_fn

    def _mock(name: str):
        def _fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            seen.append(name)
            return x

        return _fn

    try:
        S._sampler_fn = _mock  # type: ignore
        sigmas = S.seniourious_pure_scheduler(_FakeSampling(), 1)
        S.sample_seniourious_pure(_fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), disable=True)
    finally:
        S._sampler_fn = original  # type: ignore
    assert seen == ["euler"]


def test_divergence() -> None:
    """Stochastic early path differs from deterministic only path."""

    def _sde(model, x, sigmas, extra_args=None, callback=None, disable=None):
        local = x + 0.1
        s_in = local.new_ones([local.shape[0]])
        for i in range(len(sigmas) - 1):
            denoised = model(local, sigmas[i] * s_in, **(extra_args or {}))
            if float(sigmas[i + 1].item()) == 0.0:
                local = denoised
            else:
                direction = (local - denoised) / sigmas[i]
                local = local + direction * (sigmas[i + 1] - sigmas[i])
        return local

    def _ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
        local = x
        s_in = local.new_ones([local.shape[0]])
        for i in range(len(sigmas) - 1):
            denoised = model(local, sigmas[i] * s_in, **(extra_args or {}))
            if float(sigmas[i + 1].item()) == 0.0:
                local = denoised
            else:
                direction = (local - denoised) / sigmas[i]
                local = local + direction * (sigmas[i + 1] - sigmas[i])
        return local

    original = S._sampler_fn

    def _mock(name: str):
        return _sde if name == str(S.STAGE_TABLE[0]["sampler"]) else _ode

    try:
        S._sampler_fn = _mock  # type: ignore
        sigmas = torch.tensor([5.0, 3.0, 1.5, 0.7, 0.0])
        start = torch.ones(1, 2, 4, 4)
        mixed = S.sample_seniourious_pure(_fake_model, start.clone(), sigmas.clone(), disable=True)
        ode_only = _ode(_fake_model, start.clone(), sigmas.clone(), disable=True)
    finally:
        S._sampler_fn = original  # type: ignore
    assert not torch.equal(mixed, ode_only)


def test_determinism() -> None:
    """Identical inputs give identical outputs on the fixed schedule."""
    original = S._sampler_fn

    def _mock(name: str):
        def _fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
            local = x
            s_in = local.new_ones([local.shape[0]])
            for i in range(len(sigmas) - 1):
                denoised = model(local, sigmas[i] * s_in, **(extra_args or {}))
                if callback is not None:
                    callback({"x": local, "i": i, "sigma": sigmas[i], "denoised": denoised})
                if float(sigmas[i + 1].item()) == 0.0:
                    local = denoised
                else:
                    direction = (local - denoised) / sigmas[i]
                    local = local + direction * (sigmas[i + 1] - sigmas[i])
            return local

        return _fn

    try:
        S._sampler_fn = _mock  # type: ignore
        sigmas = torch.tensor([5.0, 3.0, 1.5, 0.7, 0.0])
        start = torch.ones(1, 2, 4, 4)
        first = S.sample_seniourious_pure(_fake_model, start.clone(), sigmas.clone(), disable=True)
        second = S.sample_seniourious_pure(_fake_model, start.clone(), sigmas.clone(), disable=True)
    finally:
        S._sampler_fn = original  # type: ignore
    assert torch.equal(first, second)
