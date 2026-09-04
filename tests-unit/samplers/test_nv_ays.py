"""Tests for nv ays scheduler. Synthetic data only."""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import types

import torch


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CORE_FILE = _ROOT / "nv_ays_diffusion.py"
_ADAPTER_FILE = _ROOT / "nv_ays_comfyui" / "integration.py"
_NODE_FILE = _ROOT / "custom_node" / "__init__.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


S = _load_module("nv_ays_diffusion", _CORE_FILE)

_EXPECTED_TABLE = (14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029)


class _FakeSDXL:
    """Minimal SDXL like sampling for guard tests."""

    def __init__(self) -> None:
        self.sigma_min = torch.tensor(0.0291675)
        self.sigma_max = torch.tensor(14.614642)


class _FakeSDXLFloat:
    """SDXL like sampling with plain floats."""

    def __init__(self) -> None:
        self.sigma_min = 0.0291675
        self.sigma_max = 14.615


class _FakeOther:
    """Non SDXL sampling that must be rejected."""

    def __init__(self, sigma_max) -> None:
        self.sigma_min = torch.tensor(0.002)
        self.sigma_max = torch.tensor(float(sigma_max))


def test_table_verbatim() -> None:
    """Table matches the published SDXL values exactly."""
    assert tuple(float(v) for v in S.SDXL_TABLE) == tuple(float(v) for v in _EXPECTED_TABLE)
    assert len(S.SDXL_TABLE) == 11
    assert float(S.SDXL_SIGMA_MAX) == 14.615


def test_ten_step_verbatim() -> None:
    """Ten step sigmas match the first ten table entries plus zero."""
    sigmas = S.nv_ays_sigmas_for_steps(10)
    assert len(sigmas) == 11
    vals = sigmas.tolist()
    for got, want in zip(vals[:-1], list(_EXPECTED_TABLE[:10])):
        assert abs(float(got) - float(want)) < 1e-3
    assert float(vals[-1]) == 0.0


def test_invariants_steps() -> None:
    """Length plus terminal zero plus strict decrease hold for key counts."""
    for steps in (1, 4, 10, 20, 30, 50, 100):
        sigmas = S.nv_ays_sigmas_for_steps(steps)
        assert len(sigmas) == steps + 1
        assert float(sigmas[-1].item()) == 0.0
        assert bool(torch.all(sigmas[:-1] > sigmas[1:]))
        assert sigmas.dtype == torch.float32
        assert sigmas.device.type == "cpu"


def test_single_step() -> None:
    """A single step uses the table start with zero."""
    sigmas = S.nv_ays_sigmas_for_steps(1)
    assert len(sigmas) == 2
    assert abs(float(sigmas[0].item()) - 14.615) < 1e-6
    assert float(sigmas[1].item()) == 0.0


def test_determinism() -> None:
    """Identical inputs give identical outputs."""
    for steps in (4, 10, 20, 40):
        first = S.nv_ays_sigmas_for_steps(steps)
        second = S.nv_ays_sigmas_for_steps(steps)
        assert torch.equal(first, second)


def test_no_mutation() -> None:
    """Calls never mutate the frozen table."""
    before = tuple(float(v) for v in S.SDXL_TABLE)
    for steps in (1, 4, 10, 20, 40):
        _ = S.nv_ays_sigmas_for_steps(steps)
        _ = S.loglinear_interp(list(S.SDXL_TABLE[:10]), steps if steps != 1 else 2)
    after = tuple(float(v) for v in S.SDXL_TABLE)
    assert before == after
    assert after == tuple(float(v) for v in _EXPECTED_TABLE)


def test_hierarchy() -> None:
    """Ten step knots stay a subset of twenty and forty step schedules."""
    base = S.nv_ays_sigmas_for_steps(10)[:-1].tolist()
    for steps in (20, 40):
        wide = S.nv_ays_sigmas_for_steps(steps)[:-1].tolist()
        assert len(wide) == steps
        for knot in base:
            best = min(abs(float(knot) - float(v)) for v in wide)
            assert best < 1e-6


def test_steps_below_one_raise() -> None:
    """Counts below one are rejected fail closed."""
    for bad in (0, -1, -10):
        try:
            S.nv_ays_sigmas_for_steps(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_loglinear_interp_properties() -> None:
    """Log interp keeps endpoints plus order plus determinism."""
    knots = list(_EXPECTED_TABLE[:10])
    for n in (4, 10, 20, 40):
        out = S.loglinear_interp(knots, n)
        assert len(out) == n
        assert abs(float(out[0]) - float(knots[0])) < 1e-9
        assert abs(float(out[-1]) - float(knots[-1])) < 1e-9
        assert all(a > b for a, b in zip(out, out[1:]))
        again = S.loglinear_interp(knots, n)
        assert out == again
    exact = S.loglinear_interp(knots, 10)
    for got, want in zip(exact, knots):
        assert abs(float(got) - float(want)) < 1e-9


def test_knob_free_signatures() -> None:
    """Public entry points expose only the intended knobs."""
    sig = inspect.signature(S.nv_ays_sigmas_for_steps)
    assert list(sig.parameters.keys()) == ["steps"]
    sig = inspect.signature(S.loglinear_interp)
    assert list(sig.parameters.keys()) == ["knots", "n"]
    adapter = _load_module("nv_ays_adapter_under_test", _ADAPTER_FILE)
    sig = inspect.signature(adapter.nv_ays_scheduler)
    assert list(sig.parameters.keys()) == ["model_sampling", "steps"]


def test_adapter_sdxl_pass() -> None:
    """Adapter accepts SDXL sampling and returns float CPU sigmas."""
    adapter = _load_module("nv_ays_adapter_sdxl", _ADAPTER_FILE)
    for sampling in (_FakeSDXL(), _FakeSDXLFloat()):
        for steps in (1, 4, 10, 20, 30, 50, 100):
            sigmas = adapter.nv_ays_scheduler(sampling, steps)
            assert len(sigmas) == steps + 1
            assert sigmas.dtype == torch.float32
            assert sigmas.device.type == "cpu"
            assert float(sigmas[-1].item()) == 0.0
            assert bool(torch.all(sigmas[:-1] > sigmas[1:]))


def test_adapter_matches_core() -> None:
    """Adapter output matches the frozen table output."""
    adapter = _load_module("nv_ays_adapter_match", _ADAPTER_FILE)
    for steps in (1, 4, 10, 20, 40):
        want = S.nv_ays_sigmas_for_steps(steps)
        got = adapter.nv_ays_scheduler(_FakeSDXL(), steps)
        assert torch.equal(got, want)


def test_adapter_guard_rejects_non_sdxl() -> None:
    """Adapter rejects non SDXL sigma_max fail closed."""
    adapter = _load_module("nv_ays_adapter_guard", _ADAPTER_FILE)
    for bad_max in (1.0, 7.0, 80.0, 0.5):
        try:
            adapter.nv_ays_scheduler(_FakeOther(bad_max), 10)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for non SDXL")
    for bad in (0, -2):
        try:
            adapter.nv_ays_scheduler(_FakeSDXL(), bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for bad steps")


def test_adapter_guard_missing_attr() -> None:
    """Adapter rejects sampling without sigma_max."""
    adapter = _load_module("nv_ays_adapter_missing", _ADAPTER_FILE)

    class _Empty:
        pass

    try:
        adapter.nv_ays_scheduler(_Empty(), 10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing sigma_max")


def test_registration_scheduler_only() -> None:
    """Only the scheduler is registered with use_ms true and no sampler."""
    mock_samplers = types.ModuleType("comfy.samplers")
    mock_samplers.SCHEDULER_HANDLERS = {}
    mock_samplers.SCHEDULER_NAMES = []
    mock_samplers.KSAMPLER_NAMES = ["euler", "dpmpp_2m"]
    mock_samplers.SAMPLER_NAMES = ["euler", "dpmpp_2m"]

    class _Handler:
        def __init__(self, fn, use_ms=True) -> None:
            self.handler = fn
            self.use_ms = use_ms

    mock_samplers.SchedulerHandler = _Handler
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.samplers"] = mock_samplers
    try:
        spec = importlib.util.spec_from_file_location("nv_ays_node", str(_NODE_FILE))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key in ("comfy", "comfy.samplers"):
            sys.modules.pop(key, None)
    assert "nv_ays" in mock_samplers.SCHEDULER_HANDLERS
    assert "nv_ays" in mock_samplers.SCHEDULER_NAMES
    handler = mock_samplers.SCHEDULER_HANDLERS["nv_ays"]
    assert getattr(handler, "use_ms", None) is True
    assert "nv_ays" not in mock_samplers.SAMPLER_NAMES
    assert "nv_ays" not in mock_samplers.KSAMPLER_NAMES
    assert getattr(mod, "NODE_CLASS_MAPPINGS", None) == {}
    text = pathlib.Path(_NODE_FILE).read_text()
    assert "DISCARD" not in text
    assert "k_diffusion" not in text
    assert "sample_nv_ays" not in text
