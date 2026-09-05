"""Tests for lumen geometric solver. Synthetic data only."""

from __future__ import annotations

import importlib.util
import inspect
import math
import pathlib
import sys
import types

import torch


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CORE_FILE = _ROOT / "lumen_diffusion.py"
_NODE_FILE = _ROOT / "custom_node" / "__init__.py"
_ADAPTER_FILE = _ROOT / "lumen_comfyui" / "integration.py"


def _load_core():
    spec = importlib.util.spec_from_file_location("lumen_diffusion", str(_CORE_FILE))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load_core()


def _fake_model(x, sigma, **kwargs):
    return 0.5 * x


def _karras_like(steps):
    hi = 14.6146
    lo = 0.0292
    vals = []
    # Geometric spacing with terminal zero, strictly decreasing.
    for i in range(steps):
        frac = float(i) / float(steps)
        val = math.exp(math.log(hi) * (1.0 - frac) + math.log(lo) * frac)
        vals.append(val)
    vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32)


def _linear_sigmas(steps):
    vals = [5.0 - 5.0 * float(i) / float(steps) for i in range(steps)]
    vals.append(0.0)
    vals[-2] = max(vals[-2], 0.05)
    return torch.tensor(vals, dtype=torch.float32)


def _euler_run(model, x0, sigmas, extra_args=None):
    # Plain Euler baseline for order comparisons.
    cur = x0.clone()
    n = int(cur.shape[0])
    args = extra_args or {}
    for i in range(len(sigmas) - 1):
        s = float(sigmas[i].item())
        sn = float(sigmas[i + 1].item())
        s_in = cur.new_ones([n])
        D = model(cur, sigmas[i] * s_in, **args)
        D = D.to(device=cur.device, dtype=cur.dtype)
        if sn == 0.0:
            cur = D.clone()
        else:
            rho = sn / s
            cur = rho * cur + (1.0 - rho) * D
    return cur


def _fine_reference(model, x0, sigmas, subdiv=200):
    # Fine Euler on subdivided schedule, stands in for exact truth.
    coarse = sigmas.tolist()
    fine_vals = []
    for k in range(len(coarse) - 1):
        a = float(coarse[k])
        b = float(coarse[k + 1])
        if b == 0.0:
            # Keep terminal exact, subdivide down to a small floor.
            floor = 1e-4
            for j in range(subdiv):
                frac = float(j) / float(subdiv)
                fine_vals.append(a * (1.0 - frac) + floor * frac)
            fine_vals.append(floor)
        else:
            for j in range(subdiv):
                frac = float(j) / float(subdiv)
                fine_vals.append(a * (1.0 - frac) + b * frac)
    fine_vals.append(0.0)
    fine = torch.tensor(fine_vals, dtype=torch.float64)
    x0d = x0.to(dtype=torch.float64)
    cur = _euler_run(model, x0d, fine)
    return cur.to(dtype=x0.dtype)


def test_registration_sampler_only():
    """Sampler name is registered without scheduler entries."""
    mock_samplers = types.ModuleType("comfy.samplers")
    mock_samplers.KSAMPLER_NAMES = ["euler"]
    mock_samplers.SAMPLER_NAMES = ["euler"]
    mock_sampling = types.ModuleType("comfy.k_diffusion.sampling")
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.samplers"] = mock_samplers
    sys.modules["comfy.k_diffusion"] = types.ModuleType("comfy.k_diffusion")
    sys.modules["comfy.k_diffusion.sampling"] = mock_sampling
    try:
        spec = importlib.util.spec_from_file_location("lumen_node", str(_NODE_FILE))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key in ("comfy", "comfy.samplers", "comfy.k_diffusion", "comfy.k_diffusion.sampling"):
            sys.modules.pop(key, None)
    assert "lumen" in mock_samplers.KSAMPLER_NAMES
    assert "lumen" in mock_samplers.SAMPLER_NAMES
    assert getattr(mock_sampling, "sample_lumen") is not None
    assert not hasattr(mock_samplers, "SCHEDULER_NAMES") or "lumen" not in getattr(mock_samplers, "SCHEDULER_NAMES", [])
    assert not hasattr(mock_samplers, "SCHEDULER_HANDLERS") or "lumen" not in getattr(mock_samplers, "SCHEDULER_HANDLERS", {})
    assert mod.NODE_CLASS_MAPPINGS == {}


def test_knob_free_signature():
    """Public sampler exposes the exact Comfy signature with no knobs."""
    sig = inspect.signature(S.sample_lumen)
    assert list(sig.parameters.keys()) == ["model", "x", "sigmas", "extra_args", "callback", "disable"]
    assert float(S.LUMEN_TAU) == 0.8
    src = _CORE_FILE.read_text()
    assert "DISCARD" not in src
    assert "randn" not in src
    assert "s_churn" not in src
    # Decorator keeps autograd off during sampling.
    assert getattr(S.sample_lumen, "__wrapped__", None) is not None or "no_grad" in src


def test_validate_sigmas_accepts_good():
    """Good schedules pass with length plus one and terminal zero."""
    for steps in (2, 10, 20):
        sigmas = _karras_like(steps)
        found = S.validate_sigmas(sigmas)
        assert found == steps
        assert len(sigmas) == steps + 1
        assert float(sigmas[-1].item()) == 0.0
        assert bool(torch.all(sigmas[:-1] > sigmas[1:]).item())


def test_validate_sigmas_rejects_bad():
    """Bad schedules are rejected with clear errors."""
    good = _karras_like(4)
    # Non decreasing schedule.
    bad = good.clone()
    bad[1] = bad[0]
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass
    # Missing terminal zero.
    bad = torch.tensor([5.0, 3.0, 1.0, 0.5])
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass
    # Too short.
    try:
        S.validate_sigmas(torch.tensor([1.0]))
        assert False
    except ValueError:
        pass
    # Wrong dimension.
    try:
        S.validate_sigmas(torch.zeros(2, 2))
        assert False
    except ValueError:
        pass
    # Non finite entries.
    bad = good.clone()
    bad[1] = float("nan")
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass
    bad = good.clone()
    bad[2] = float("inf")
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass


def test_count_nfe_equals_steps():
    """Evaluations equal steps with one call per step."""
    for steps in (1, 4, 10, 20):
        assert S.count_nfe(steps) == steps
    try:
        S.count_nfe(0)
        assert False
    except ValueError:
        pass


def test_nfe_matches_model_calls():
    """Mock model is called once per step."""
    calls = {"n": 0}

    def _counting(x, sigma, **kwargs):
        calls["n"] += 1
        return 0.5 * x

    for steps in (4, 10):
        calls["n"] = 0
        sigmas = _karras_like(steps)
        start = torch.ones(1, 2, 4, 4)
        S.sample_lumen(_counting, start.clone(), sigmas.clone(), disable=True)
        assert calls["n"] == steps
        assert calls["n"] == S.count_nfe(steps)


def test_callback_called_each_step():
    """Callback runs once per step with global indices."""
    for steps in (4, 10):
        seen = []

        def _cb(data):
            seen.append(int(data["i"]))

        sigmas = _karras_like(steps)
        S.sample_lumen(_fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), callback=_cb, disable=True)
        assert seen == list(range(steps))


def test_determinism():
    """Identical inputs give identical outputs."""
    sigmas = _karras_like(10)
    start = torch.ones(1, 2, 4, 4) * 0.7
    first = S.sample_lumen(_fake_model, start.clone(), sigmas.clone(), disable=True)
    second = S.sample_lumen(_fake_model, start.clone(), sigmas.clone(), disable=True)
    assert torch.equal(first, second)


def test_dtype_device_batch_preserved():
    """Output keeps dtype and batch shape of the input."""
    sigmas = _karras_like(6)
    start = torch.ones(2, 3, 4, 4, dtype=torch.float32) * 0.5
    out = S.sample_lumen(_fake_model, start.clone(), sigmas.clone(), disable=True)
    assert out.dtype == start.dtype
    assert out.device == start.device
    assert tuple(out.shape) == tuple(start.shape)
    start64 = torch.ones(1, 2, 4, 4, dtype=torch.float64) * 0.3
    out64 = S.sample_lumen(_fake_model, start64.clone(), sigmas.clone(), disable=True)
    assert out64.dtype == torch.float64


def test_terminal_zero_returns_denoised():
    """Terminal step returns denoised prediction directly."""
    D = torch.ones(1, 2, 3, 3) * 2.0
    x = torch.ones(1, 2, 3, 3) * 5.0
    nxt, kappa = S.lumen_step(x, 0.5, 0.0, D, torch.zeros_like(D), 1.0)
    assert torch.equal(nxt, D)
    assert float(kappa) == 1.0
    # Full run also ends at denoised for a constant model.
    def _const(x_, sigma, **kwargs):
        return torch.ones_like(x_) * 1.5

    sigmas = torch.tensor([3.0, 1.0, 0.0])
    out = S.sample_lumen(_const, torch.zeros(1, 1, 2, 2), sigmas, disable=True)
    assert bool(torch.allclose(out, torch.ones_like(out) * 1.5))


def test_first_step_matches_euler():
    """First step without history equals Euler."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4)
    D = torch.randn(1, 2, 4, 4)
    nxt, kappa = S.lumen_step(x, 5.0, 3.0, D, None, None)
    ref = S.euler_step(x, 5.0, 3.0, D)
    assert bool(torch.allclose(nxt, ref))
    assert float(kappa) == 1.0


def test_order_leak_synthetic():
    """LUMEN beats Euler on a leaky probe that drifts with log sigma."""

    def _leak(x, sigma, **kwargs):
        # D drifts linearly in log sigma, Euler holds it constant.
        s0 = float(sigma.view(-1)[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        drift = 0.6 * math.log(max(s0, 1e-4)) + 0.4
        return 0.5 * x + drift * torch.ones_like(x) * 0.3

    for steps in (10, 20):
        sigmas = _karras_like(steps)
        start = torch.ones(1, 2, 8, 8) * 0.8
        ref = _fine_reference(_leak, start.clone(), sigmas)
        got = S.sample_lumen(_leak, start.clone(), sigmas.clone(), disable=True)
        base = _euler_run(_leak, start.clone(), sigmas.clone())
        err_got = float(torch.mean((got - ref) ** 2).item())
        err_base = float(torch.mean((base - ref) ** 2).item())
        assert err_got < err_base


def test_order_osc_synthetic():
    """LUMEN beats Euler on an oscillatory probe in log sigma."""

    def _osc(x, sigma, **kwargs):
        s0 = float(sigma.view(-1)[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        wob = math.sin(1.0 * math.log(max(s0, 1e-4))) * 0.5
        return 0.4 * x + wob * torch.ones_like(x) * 0.4 + 0.2

    for steps in (10, 20):
        sigmas = _karras_like(steps)
        start = torch.ones(1, 2, 8, 8) * 0.6
        ref = _fine_reference(_osc, start.clone(), sigmas)
        got = S.sample_lumen(_osc, start.clone(), sigmas.clone(), disable=True)
        base = _euler_run(_osc, start.clone(), sigmas.clone())
        err_got = float(torch.mean((got - ref) ** 2).item())
        err_base = float(torch.mean((base - ref) ** 2).item())
        assert err_got < err_base


def test_guards_nan_inf_fail_closed():
    """Non finite model outputs and sigmas fail closed."""

    def _nan(x, sigma, **kwargs):
        return torch.full_like(x, float("nan"))

    def _inf(x, sigma, **kwargs):
        return torch.full_like(x, float("inf"))

    sigmas = _karras_like(4)
    start = torch.ones(1, 2, 4, 4)
    for bad_model in (_nan, _inf):
        try:
            S.sample_lumen(bad_model, start.clone(), sigmas.clone(), disable=True)
            assert False
        except RuntimeError:
            pass
    bad = sigmas.clone()
    bad[1] = float("nan")
    try:
        S.sample_lumen(_fake_model, start.clone(), bad, disable=True)
        assert False
    except ValueError:
        pass
    # Shape mismatch is rejected as well.
    def _wrong_shape(x, sigma, **kwargs):
        return torch.ones(1, 3, 4, 4)

    try:
        S.sample_lumen(_wrong_shape, start.clone(), sigmas.clone(), disable=True)
        assert False
    except ValueError:
        pass


def test_flow_v_fail_closed():
    """Flow and velocity hints are rejected before sampling."""

    class _Flow:
        model_type = "flow"

        def __call__(self, x, sigma, **kwargs):
            return 0.5 * x

    class _Vel:
        prediction_type = "velocity"

        def __call__(self, x, sigma, **kwargs):
            return 0.5 * x

    sigmas = _karras_like(4)
    start = torch.ones(1, 2, 4, 4)
    for bad in (_Flow(), _Vel()):
        try:
            S.sample_lumen(bad, start.clone(), sigmas.clone(), disable=True)
            assert False
        except ValueError:
            pass


def test_damping_bounds_and_scheduler_invariant():
    """Kappa stays bounded and any schedule shape works."""
    a = torch.ones(1, 2, 4, 4)
    b = torch.ones(1, 2, 4, 4) * 3.0
    k = S.damping_factor(a, b)
    assert 0.0 <= float(k) <= 1.0
    k2 = S.damping_factor(a, a.clone())
    assert 0.0 <= float(k2) <= 1.0
    # Two schedule shapes both finish finite and deterministic.
    for sigmas in (_karras_like(10), _linear_sigmas(10)):
        s1 = S.sample_lumen(_fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), disable=True)
        s2 = S.sample_lumen(_fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), disable=True)
        assert bool(torch.isfinite(s1).all().item())
        assert torch.equal(s1, s2)


def test_adapter_has_no_scheduler():
    """Adapter stub exposes only the sampler."""
    text = _ADAPTER_FILE.read_text()
    assert "sample_lumen" in text
    assert "SCHEDULER" not in text
    assert "SchedulerHandler" not in text
    assert "seniourious" not in text.lower()
    spec = importlib.util.spec_from_file_location("lumen_adapter", str(_ADAPTER_FILE))
    assert spec is not None and spec.loader is not None


def test_frozen_polish_constants():
    """Tail and guard thresholds stay frozen with no extra knobs."""
    assert int(S.LUMEN_TAIL_EULER) == 2
    assert float(S.LUMEN_TAU) == 0.8
    ratio = float(S.LUMEN_GUARD_RATIO)
    assert 0.3 <= ratio <= 0.5
    assert "LUMEN_TAIL_EULER" in S.__all__
    assert "LUMEN_GUARD_RATIO" in S.__all__
    sig = inspect.signature(S.sample_lumen)
    assert list(sig.parameters.keys()) == ["model", "x", "sigmas", "extra_args", "callback", "disable"]
    src = _CORE_FILE.read_text()
    assert "DISCARD" not in src
    assert "randn" not in src
    assert "s_churn" not in src


def test_tail_euler_forces_euler():
    """Two non-terminal steps before terminal stay Euler."""
    torch.manual_seed(11)
    x = torch.randn(1, 2, 4, 4)
    D_i = torch.randn(1, 2, 4, 4)
    D_prev = D_i * 1.02 + 0.01
    s_prev, s, sn = 5.0, 3.0, 2.0
    ref = S.euler_step(x, s, sn, D_i)
    # Interior index keeps the geometric correction.
    mid, _ = S.lumen_step(x, s, sn, D_i, D_prev, s_prev, 3, 10)
    assert not bool(torch.allclose(mid, ref))
    # Tail indices fall back to Euler with the same inputs.
    for idx in (7, 8):
        nxt, kappa = S.lumen_step(x, s, sn, D_i, D_prev, s_prev, idx, 10)
        assert bool(torch.allclose(nxt, ref))
        assert float(kappa) == 1.0
    # Without indices the same inputs keep the correction.
    plain, _ = S.lumen_step(x, s, sn, D_i, D_prev, s_prev)
    assert bool(torch.allclose(plain, mid))


def test_tail_terminal_still_exact():
    """Terminal step returns denoised exactly even with tail indices."""
    D = torch.ones(1, 2, 3, 3) * 2.0
    x = torch.ones(1, 2, 3, 3) * 5.0
    prev = torch.zeros_like(D)
    for idx in (8, 9):
        nxt, kappa = S.lumen_step(x, 0.5, 0.0, D, prev, 1.0, idx, 10)
        assert torch.equal(nxt, D)
        assert float(kappa) == 1.0
    # Full polished run with a constant model still ends at denoised.
    def _const(x_, sigma, **kwargs):
        return torch.ones_like(x_) * 1.5

    sigmas = torch.tensor([3.0, 1.0, 0.0])
    out = S.sample_lumen(_const, torch.zeros(1, 1, 2, 2), sigmas, disable=True)
    assert bool(torch.allclose(out, torch.ones_like(out) * 1.5))


def test_guard_smooth_preserves_correction():
    """Smooth history keeps the correction and preserves gain."""
    torch.manual_seed(7)
    x = torch.ones(1, 2, 4, 4) * 0.5
    D_prev = torch.ones(1, 2, 4, 4) * 1.01
    D_i = torch.ones(1, 2, 4, 4) * 1.0
    s_prev, s, sn = 1.5, 1.0, 0.7
    ref = S.euler_step(x, s, sn, D_i)
    nxt, kappa = S.lumen_step(x, s, sn, D_i, D_prev, s_prev, 2, 10)
    assert not bool(torch.allclose(nxt, ref))
    # Polished sampler still beats Euler on the smooth leak probe.
    def _leak(x_, sigma, **kwargs):
        s0 = float(sigma.view(-1)[0].item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        drift = 0.6 * math.log(max(s0, 1e-4)) + 0.4
        return 0.5 * x_ + drift * torch.ones_like(x_) * 0.3

    sigmas = _karras_like(10)
    start = torch.ones(1, 2, 8, 8) * 0.8
    ref_trajectory = _fine_reference(_leak, start.clone(), sigmas)
    got = S.sample_lumen(_leak, start.clone(), sigmas.clone(), disable=True)
    base = _euler_run(_leak, start.clone(), sigmas.clone())
    err_got = float(torch.mean((got - ref_trajectory) ** 2).item())
    err_base = float(torch.mean((base - ref_trajectory) ** 2).item())
    assert err_got < err_base


def test_guard_flip_falls_back():
    """Large correction relative to the Euler step falls back to Euler."""
    x = torch.ones(1, 2, 4, 4) * 0.5
    D_prev = torch.ones(1, 2, 4, 4) * -2.0
    D_i = torch.ones(1, 2, 4, 4) * 2.0
    s_prev, s, sn = 1.5, 1.0, 0.7
    ref = S.euler_step(x, s, sn, D_i)
    nxt, kappa = S.lumen_step(x, s, sn, D_i, D_prev, s_prev, 2, 10)
    assert bool(torch.allclose(nxt, ref))
    assert float(kappa) == 1.0
    # Zero-mean checker flip also falls back while staying finite.
    B, C, H, W = 1, 1, 8, 8
    xx = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W).expand(B, C, H, W)
    yy = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1).expand(B, C, H, W)
    checker = ((xx + yy) % 2) * 2 - 1
    Dc_prev = 0.6 + 0.3 * checker
    Dc_i = 0.6 - 0.3 * checker
    xc = torch.ones(B, C, H, W) * 0.5
    ref_c = S.euler_step(xc, s, sn, Dc_i)
    nxt_c, _ = S.lumen_step(xc, s, sn, Dc_i, Dc_prev, s_prev, 2, 10)
    assert bool(torch.allclose(nxt_c, ref_c))
    assert bool(torch.isfinite(nxt_c).all().item())


def test_polished_nfe_determinism():
    """Polished sampler keeps NFE, determinism, dtype, and callbacks."""
    calls = {"n": 0}

    def _counting(x, sigma, **kwargs):
        calls["n"] += 1
        return 0.5 * x

    for steps in (4, 10):
        sigmas = _karras_like(steps)
        start = torch.ones(1, 2, 4, 4) * 0.7
        calls["n"] = 0
        first = S.sample_lumen(_counting, start.clone(), sigmas.clone(), disable=True)
        assert calls["n"] == steps
        assert calls["n"] == S.count_nfe(steps)
        calls["n"] = 0
        second = S.sample_lumen(_counting, start.clone(), sigmas.clone(), disable=True)
        assert calls["n"] == steps
        assert torch.equal(first, second)
        seen = []

        def _cb(data):
            seen.append(int(data["i"]))

        S.sample_lumen(_fake_model, start.clone(), sigmas.clone(), callback=_cb, disable=True)
        assert seen == list(range(steps))
    start64 = torch.ones(1, 2, 4, 4, dtype=torch.float64) * 0.3
    out64 = S.sample_lumen(_fake_model, start64.clone(), _karras_like(6).clone(), disable=True)
    assert out64.dtype == torch.float64
