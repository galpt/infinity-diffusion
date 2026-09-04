"""Tests for era solver sampler. Synthetic data only."""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import types

import torch


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CORE_FILE = _ROOT / "era_solver_diffusion.py"
_NODE_FILE = _ROOT / "custom_node" / "__init__.py"
_STUB_FILE = _ROOT / "era_solver_comfyui" / "__init__.py"
_README_FILE = _ROOT / "README.md"


def _load_core():
    spec = importlib.util.spec_from_file_location("era_solver_diffusion", str(_CORE_FILE))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load_core()


def _fake_model(x, sigma, **kwargs):
    return 0.5 * x


def _make_sigmas(steps):
    vals = [float(5.0 - float(i) * 0.4) for i in range(int(steps))]
    vals = [float(v) if float(v) > 0.5 else 0.5 for v in vals]
    vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32)


def test_registration_sampler_only():
    """Sampler registers alone with no scheduler side effects."""
    mock_samplers = types.ModuleType("comfy.samplers")
    mock_samplers.SCHEDULER_HANDLERS = {}
    mock_samplers.SCHEDULER_NAMES = []
    mock_samplers.KSAMPLER_NAMES = ["euler", "dpmpp_2m"]
    mock_samplers.SAMPLER_NAMES = ["euler", "dpmpp_2m"]
    mock_sampling = types.ModuleType("comfy.k_diffusion.sampling")
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.samplers"] = mock_samplers
    sys.modules["comfy.k_diffusion"] = types.ModuleType("comfy.k_diffusion")
    sys.modules["comfy.k_diffusion.sampling"] = mock_sampling
    try:
        spec = importlib.util.spec_from_file_location("era_solver_node", str(_NODE_FILE))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key in ("comfy", "comfy.samplers", "comfy.k_diffusion", "comfy.k_diffusion.sampling"):
            sys.modules.pop(key, None)
    assert "era_solver" in mock_samplers.KSAMPLER_NAMES
    assert "era_solver" in mock_samplers.SAMPLER_NAMES
    assert "era_solver" not in mock_samplers.SCHEDULER_NAMES
    assert "era_solver" not in mock_samplers.SCHEDULER_HANDLERS
    assert getattr(mock_sampling, "sample_era_solver") is not None
    assert getattr(mod, "NODE_CLASS_MAPPINGS", None) == {}
    text = pathlib.Path(_NODE_FILE).read_text()
    assert "DISCARD" not in text
    assert "SCHEDULER" not in text
    assert "SchedulerHandler" not in text
    core_text = pathlib.Path(_CORE_FILE).read_text()
    assert "DISCARD" not in core_text
    assert "randn" not in core_text
    assert "random" not in core_text.lower() or "error robust" in core_text.lower()


def test_knob_free_signatures():
    """Public entry points expose only the intended inputs."""
    sig = inspect.signature(S.sample_era_solver)
    assert list(sig.parameters.keys()) == ["model", "x", "sigmas", "extra_args", "callback", "disable"]
    sig = inspect.signature(S.count_nfe)
    assert list(sig.parameters.keys()) == ["steps"]
    sig = inspect.signature(S.validate_sigmas)
    assert list(sig.parameters.keys()) == ["sigmas"]
    for name in ["_sigma_to_alpha", "_sigma_to_logsnr", "_lagrange_weights", "_select_indices", "_proxy_error", "_ddim_step", "_eps_from_denoised"]:
        assert hasattr(S, str(name))


def test_frozen_constants():
    """Frozen order and scale stay at paper values with correct coeffs."""
    assert int(S.FROZEN_K) == 4
    assert float(S.FROZEN_LAMBDA) == 10.0
    sig = inspect.signature(S._select_indices)
    assert list(sig.parameters.keys()) == ["i", "delta"]


def test_validate_sigmas():
    """Valid sigmas pass and malformed sigmas raise fail closed."""
    good = torch.tensor([5.0, 3.0, 1.5, 0.7, 0.0])
    assert int(S.validate_sigmas(good)) == 4
    bad_cases = [
        torch.tensor([5.0, 3.0, 1.5, 0.7]),
        torch.tensor([5.0, 3.0, 3.0, 0.0]),
        torch.tensor([5.0, 3.0, 1.0, 0.2]),
        torch.tensor([0.0]),
        torch.tensor([5.0, float("nan"), 0.0]),
        torch.tensor([5.0, float("inf"), 0.0]),
        torch.tensor([-1.0, -2.0, 0.0]),
    ]
    for bad in bad_cases:
        try:
            S.validate_sigmas(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    try:
        S.validate_sigmas(torch.tensor([[5.0, 0.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_count_nfe():
    """Each step costs one evaluation so total equals steps."""
    for steps in (1, 2, 4, 10, 20):
        assert int(S.count_nfe(int(steps))) == int(steps)
    for bad in (0, -1, -5):
        try:
            S.count_nfe(int(bad))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_lagrange_weights():
    """Lagrange weights sum to one with fp32 rounding control."""
    w = S._lagrange_weights(0.5, [1.0, 0.0, -1.0, -2.0])
    assert len(w) == 4
    assert abs(float(sum(float(v) for v in w)) - 1.0) < 1e-6
    again = S._lagrange_weights(0.5, [1.0, 0.0, -1.0, -2.0])
    assert w == again
    w2 = S._lagrange_weights(1.0, [1.0, 0.0, -1.0, -2.0])
    assert abs(float(w2[0]) - 1.0) < 1e-6
    assert abs(float(w2[1])) < 1e-6
    w3 = S._lagrange_weights(2.0, [2.0])
    assert w3 == [1.0]
    try:
        S._lagrange_weights(0.0, [1.0, 1.0, 0.0, -1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_select_indices():
    """Warped indices stay distinct sorted and keep current."""
    for i in (3, 5, 10, 20):
        for delta in (0.0, 1.0, 10.0, 50.0):
            picked = S._select_indices(int(i), float(delta))
            assert len(picked) == 4
            assert picked == sorted(picked)
            assert len(set(picked)) == 4
            assert min(picked) >= 0 and max(picked) == int(i)
    near = S._select_indices(10, 0.0)
    assert near == [7, 8, 9, 10]
    far = S._select_indices(10, 100.0)
    assert far[-1] == 10
    assert far[0] <= 1
    try:
        S._select_indices(2, 10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_proxy_error():
    """Proxy error is squared L2 with clamp and fail closed."""
    a = torch.ones(1, 2, 4, 4)
    b = torch.ones(1, 2, 4, 4)
    assert float(S._proxy_error(a, b)) == 0.0
    c = torch.zeros(1, 2, 4, 4)
    val = float(S._proxy_error(a, c))
    assert val > 0.0 and val <= 100.0
    big_pred = torch.full((1, 2, 4, 4), 1e4)
    big_obs = torch.zeros(1, 2, 4, 4)
    assert float(S._proxy_error(big_pred, big_obs)) == 100.0
    try:
        S._proxy_error(torch.full((1, 2, 2, 2), float("nan")), torch.zeros(1, 2, 2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_eps_ddim_helpers():
    """Alpha plus log SNR plus eps plus DDIM behave sanely."""
    assert abs(float(S._sigma_to_alpha(0.0)) - 1.0) < 1e-9
    assert float(S._sigma_to_alpha(1.0)) < float(S._sigma_to_alpha(0.0))
    assert float(S._sigma_to_logsnr(0.5)) > float(S._sigma_to_logsnr(2.0))
    assert float(S._sigma_to_logsnr(0.0)) > float(S._sigma_to_logsnr(0.1))
    x = torch.ones(1, 2, 4, 4)
    denoised = 0.5 * x
    eps = S._eps_from_denoised(x, denoised, 2.0)
    assert torch.allclose(eps, torch.full_like(x, 0.25))
    out = S._ddim_step(x, eps, 2.0, 1.0)
    assert tuple(out.shape) == tuple(x.shape)
    assert bool(torch.isfinite(out).all().item())
    term = S._ddim_step(x, eps, 2.0, 0.0)
    assert torch.allclose(term, denoised)
    try:
        S._eps_from_denoised(x, denoised, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_determinism_nfe():
    """Identical inputs give identical outputs with one call per step."""
    counts = {"n": 0}

    def _counting(x, sigma, **kwargs):
        counts["n"] += 1
        return 0.5 * x

    for steps in (1, 4, 10):
        sigmas = _make_sigmas(int(steps))
        counts["n"] = 0
        first = S.sample_era_solver(_counting, torch.ones(1, 2, 4, 4), sigmas.clone(), extra_args={}, disable=True)
        assert counts["n"] == int(steps)
        assert int(S.count_nfe(int(steps))) == int(steps)
        counts["n"] = 0
        second = S.sample_era_solver(_counting, torch.ones(1, 2, 4, 4), sigmas.clone(), extra_args={}, disable=True)
        assert counts["n"] == int(steps)
        assert torch.equal(first, second)
    sigmas = torch.tensor([2.0, 1.0, 0.0])
    out = S.sample_era_solver(_fake_model, torch.ones(1, 1, 2, 2), sigmas.clone(), extra_args={}, disable=True)
    assert tuple(out.shape) == (1, 1, 2, 2)


def test_extra_args_callback_failclosed_preservation():
    """Identity plus callbacks plus guards plus preservation hold."""
    seen_kwargs = []
    seen_ids = []

    def _spy(x, sigma, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return 0.5 * x

    sigmas = _make_sigmas(6)
    given = {"cfg": 1.0}
    before_id = id(given)
    seen_idx = []

    def _cb(data):
        seen_idx.append(int(data["i"]))

    start = torch.ones(2, 3, 4, 4, dtype=torch.float32)
    out = S.sample_era_solver(_spy, start.clone(), sigmas.clone(), extra_args=given, callback=_cb, disable=True)
    assert id(given) == before_id
    assert len(seen_kwargs) == 6
    for kw in seen_kwargs:
        assert kw == {"cfg": 1.0}
    assert seen_idx == list(range(6))
    assert tuple(out.shape) == tuple(start.shape)
    assert out.dtype == start.dtype
    assert str(out.device) == str(start.device)
    small = torch.tensor([1.0, 0.5, 0.0])
    out_small = S.sample_era_solver(_fake_model, torch.ones(1, 2, 2, 2), small.clone(), extra_args={}, disable=True)
    assert bool(torch.isfinite(out_small).all().item())
    term_sigmas = torch.tensor([2.0, 1.0, 0.0])
    term_out = S.sample_era_solver(_fake_model, torch.ones(1, 1, 2, 2), term_sigmas.clone(), extra_args={}, disable=True)
    assert bool(torch.isfinite(term_out).all().item())
    try:
        S.sample_era_solver(_fake_model, torch.full((1, 2, 2, 2), float("nan")), term_sigmas.clone(), extra_args={}, disable=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    class _Flow:
        model_type = "flow"

        def __call__(self, x, sigma, **kwargs):
            return 0.5 * x

    try:
        S.sample_era_solver(_Flow(), torch.ones(1, 2, 2, 2), term_sigmas.clone(), extra_args={}, disable=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    try:
        S.sample_era_solver(_fake_model, torch.ones(1, 2, 2, 2), term_sigmas.clone(), extra_args={"model_type": "v_prediction"}, disable=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    stub_text = pathlib.Path(_STUB_FILE).read_text()
    assert "scheduler" not in stub_text.lower()
    readme_text = pathlib.Path(_README_FILE).read_text()
    assert "https://arxiv.org/abs/2301.12935" in readme_text
    assert "era_solver" in readme_text
