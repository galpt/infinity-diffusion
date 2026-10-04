"""Tests for Euler-M ancestral SDE sampler. Synthetic data only."""

from __future__ import annotations

import importlib.util
import inspect
import math
import pathlib
import sys
import types

import torch


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CORE_FILE = _ROOT / "euler_m_diffusion.py"
_NODE_FILE = _ROOT / "custom_node" / "__init__.py"
_ADAPTER_FILE = _ROOT / "euler_m_comfyui" / "integration.py"


def _load_core():
    spec = importlib.util.spec_from_file_location("euler_m_diffusion", str(_CORE_FILE))
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


def _rf_sigmas(steps):
    vals = [1.0 - float(i) / float(steps) * 0.95 for i in range(steps)]
    vals.append(0.0)
    vals[-2] = max(vals[-2], 0.05)
    return torch.tensor(vals, dtype=torch.float32)


def _euler_run(model, x0, sigmas, extra_args=None):
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


def _ancestral_manual(model, x0, sigmas, eta=1.0, s_noise=1.0, noise_fn=None, extra_args=None):
    # Independent ancestral algebra using the same split as Comfy helpers.
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
            continue
        if not eta:
            sigma_down, sigma_up = sn, 0.0
        else:
            sigma_up = min(sn, eta * (sn**2 * (s**2 - sn**2) / s**2) ** 0.5)
            sigma_down = (sn**2 - sigma_up**2) ** 0.5
        d = (cur - D) / s
        dt = sigma_down - s
        cur = cur + d * dt
        if eta:
            noise = noise_fn(cur.shape) if noise_fn is not None else torch.zeros_like(cur)
            cur = cur + noise.to(device=cur.device, dtype=cur.dtype) * s_noise * sigma_up
    return cur


def test_registration_sampler_only():
    """Euler-M name is registered without scheduler entries."""
    mock_samplers = types.ModuleType("comfy.samplers")
    mock_samplers.KSAMPLER_NAMES = ["euler"]
    mock_samplers.SAMPLER_NAMES = ["euler"]
    mock_sampling = types.ModuleType("comfy.k_diffusion.sampling")
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.samplers"] = mock_samplers
    sys.modules["comfy.k_diffusion"] = types.ModuleType("comfy.k_diffusion")
    sys.modules["comfy.k_diffusion.sampling"] = mock_sampling
    try:
        spec = importlib.util.spec_from_file_location("euler_m_node", str(_NODE_FILE))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key in ("comfy", "comfy.samplers", "comfy.k_diffusion", "comfy.k_diffusion.sampling"):
            sys.modules.pop(key, None)
    assert "euler_m" in mock_samplers.KSAMPLER_NAMES
    assert "euler_m" in mock_samplers.SAMPLER_NAMES
    assert getattr(mock_sampling, "sample_euler_m") is not None
    assert not hasattr(mock_samplers, "SCHEDULER_NAMES") or "euler_m" not in getattr(
        mock_samplers, "SCHEDULER_NAMES", []
    )
    assert not hasattr(mock_samplers, "SCHEDULER_HANDLERS") or "euler_m" not in getattr(
        mock_samplers, "SCHEDULER_HANDLERS", {}
    )
    assert mod.NODE_CLASS_MAPPINGS == {}
    src = _NODE_FILE.read_text()
    assert "_NAME" in src
    assert "euler_m" in src


def test_signature_defaults():
    """Sampler exposes Comfy signature plus eta, s_noise, noise_sampler."""
    for fn_name in ("sample_euler_m", "sample_euler_m_RF"):
        fn = getattr(S, fn_name)
        sig = inspect.signature(fn)
        assert list(sig.parameters.keys()) == [
            "model",
            "x",
            "sigmas",
            "extra_args",
            "callback",
            "disable",
            "eta",
            "s_noise",
            "noise_sampler",
        ]
        assert float(sig.parameters["eta"].default) == 1.0
        assert float(sig.parameters["s_noise"].default) == 1.0
        assert sig.parameters["noise_sampler"].default is None
    src = _CORE_FILE.read_text()
    assert getattr(S.sample_euler_m, "__wrapped__", None) is not None or "no_grad" in src


def test_no_forbidden_strings():
    """Core and adapter stay sampler-only with no tree or churn controls."""
    core = _CORE_FILE.read_text()
    for bad in ("BrownianTree", "s_churn", "DISCARD"):
        assert bad not in core
    for bad in ("SCHEDULER", "SchedulerHandler"):
        assert bad not in core
        assert bad not in _ADAPTER_FILE.read_text()
        assert bad not in _NODE_FILE.read_text()


def test_adapter_reexport_only():
    """Adapter reexports the sampler pair with no extra logic."""
    text = _ADAPTER_FILE.read_text()
    assert "sample_euler_m" in text
    assert "sample_euler_m_RF" in text
    assert "euler_m_diffusion" in text


def test_validate_sigmas_accepts_good():
    """Good schedules pass with tolerant terminal handling."""
    for steps in (2, 10, 20):
        sigmas = _karras_like(steps)
        assert S.validate_sigmas(sigmas) == steps
        assert S.validate_sigmas(sigmas, steps) == steps
        assert S.validate_sigmas(_linear_sigmas(steps)) == steps
        assert S.validate_sigmas(_rf_sigmas(steps)) == steps


def test_validate_tolerant_terminal():
    """Near-zero terminal within tolerance passes, far values fail."""
    assert S.is_terminal_sigma(torch.tensor(0.0))
    assert S.is_terminal_sigma(0.0)
    assert S.is_terminal_sigma(torch.tensor(1e-12))
    assert not S.is_terminal_sigma(torch.tensor(1e-5))
    good = _karras_like(4)
    near = good.clone()
    near[-1] = 1e-12
    assert S.validate_sigmas(near) == 4
    bad = good.clone()
    bad[-1] = 1e-5
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass


def test_validate_sigmas_rejects_bad():
    """Bad schedules are rejected with clear errors."""
    good = _karras_like(4)
    bad = good.clone()
    bad[1] = bad[0]
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass
    bad = torch.tensor([5.0, 3.0, 1.0, 0.5])
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass
    try:
        S.validate_sigmas(torch.tensor([1.0]))
        assert False
    except ValueError:
        pass
    try:
        S.validate_sigmas(torch.zeros(2, 2))
        assert False
    except ValueError:
        pass
    bad = good.clone()
    bad[1] = float("nan")
    try:
        S.validate_sigmas(bad)
        assert False
    except ValueError:
        pass
    try:
        S.validate_sigmas(good, steps=99)
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
        S.sample_euler_m(_counting, start.clone(), sigmas.clone(), disable=True, eta=0.0)
        assert calls["n"] == steps
        assert calls["n"] == S.count_nfe(steps)


def test_eta_zero_equals_euler():
    """With eta zero the SDE collapses exactly to the Euler ODE."""
    for sigmas in (_karras_like(10), _linear_sigmas(10)):
        start = torch.ones(1, 2, 4, 4) * 0.7
        got = S.sample_euler_m(
            _fake_model,
            start.clone(),
            sigmas.clone(),
            disable=True,
            eta=0.0,
            noise_sampler=lambda a, b: torch.zeros_like(start),
        )
        ref = _euler_run(_fake_model, start.clone(), sigmas.clone())
        assert bool(torch.allclose(got, ref, atol=1e-6))
    # Single step helper also matches Euler at eta zero.
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4)
    D = torch.randn(1, 2, 4, 4)
    nxt = S.euler_m_step(x, 5.0, 3.0, D, None, eta=0.0)
    rho = 3.0 / 5.0
    assert bool(torch.allclose(nxt, rho * x + (1.0 - rho) * D))


def test_eta_one_matches_ancestral_algebra():
    """With eta one the update equals drift plus reinjected noise."""
    torch.manual_seed(3)
    sigmas = _karras_like(6)
    start = torch.randn(1, 2, 4, 4)
    fixed = torch.randn(1, 2, 4, 4) * 0.7

    def _fixed_sampler(a, b):
        return fixed.clone()

    got = S.sample_euler_m(
        _fake_model, start.clone(), sigmas.clone(), disable=True, eta=1.0, s_noise=1.0, noise_sampler=_fixed_sampler
    )
    ref = _ancestral_manual(
        _fake_model, start.clone(), sigmas.clone(), eta=1.0, s_noise=1.0, noise_fn=lambda shape: fixed.clone()
    )
    assert bool(torch.allclose(got, ref, atol=1e-6))
    # s_noise scales the reinjected part.
    got_half = S.sample_euler_m(
        _fake_model, start.clone(), sigmas.clone(), disable=True, eta=1.0, s_noise=0.5, noise_sampler=_fixed_sampler
    )
    ref_half = _ancestral_manual(
        _fake_model, start.clone(), sigmas.clone(), eta=1.0, s_noise=0.5, noise_fn=lambda shape: fixed.clone()
    )
    assert bool(torch.allclose(got_half, ref_half, atol=1e-6))
    assert not bool(torch.allclose(got, got_half))


def test_determinism_and_seed_sensitivity():
    """Fixed sampler is deterministic, default sampler reacts to seed."""
    sigmas = _karras_like(6)
    start = torch.ones(1, 2, 4, 4) * 0.6
    fixed = lambda a, b: torch.ones(1, 2, 4, 4) * 0.25  # noqa: E731
    first = S.sample_euler_m(_fake_model, start.clone(), sigmas.clone(), disable=True, noise_sampler=fixed)
    second = S.sample_euler_m(_fake_model, start.clone(), sigmas.clone(), disable=True, noise_sampler=fixed)
    assert torch.equal(first, second)
    # Seeded default sampler is deterministic per seed and sensitive across seeds.
    seeded_a = S.sample_euler_m(
        _fake_model, start.clone(), sigmas.clone(), disable=True, extra_args={"seed": 7}
    )
    seeded_b = S.sample_euler_m(
        _fake_model, start.clone(), sigmas.clone(), disable=True, extra_args={"seed": 7}
    )
    seeded_c = S.sample_euler_m(
        _fake_model, start.clone(), sigmas.clone(), disable=True, extra_args={"seed": 8}
    )
    assert torch.equal(seeded_a, seeded_b)
    assert not torch.equal(seeded_a, seeded_c)


def test_callback_called_each_step():
    """Callback runs once per step with the Comfy payload."""
    for steps in (4, 10):
        seen = []
        payloads = []

        def _cb(data):
            seen.append(int(data["i"]))
            payloads.append(data)

        sigmas = _karras_like(steps)
        S.sample_euler_m(_fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), callback=_cb, disable=True)
        assert seen == list(range(steps))
        for data in payloads:
            assert set(("x", "i", "sigma", "sigma_hat", "denoised")) <= set(data.keys())


def test_terminal_zero_returns_denoised():
    """Terminal step returns denoised prediction directly."""

    def _const(x, sigma, **kwargs):
        return torch.ones_like(x) * 1.5

    sigmas = torch.tensor([3.0, 1.0, 0.0])
    out = S.sample_euler_m(_const, torch.zeros(1, 1, 2, 2), sigmas, disable=True)
    assert bool(torch.allclose(out, torch.ones_like(out) * 1.5))
    # Near-zero terminal within tolerance behaves the same.
    near = torch.tensor([3.0, 1.0, 1e-12])
    out_near = S.sample_euler_m(_const, torch.zeros(1, 1, 2, 2), near, disable=True)
    assert bool(torch.allclose(out_near, torch.ones_like(out_near) * 1.5))
    # Step helper terminal path clones denoised.
    D = torch.ones(1, 2, 3, 3) * 2.0
    x = torch.ones(1, 2, 3, 3) * 5.0
    assert torch.equal(S.euler_m_step(x, 0.5, 0.0, D, None, eta=1.0), D)


def test_rf_parity():
    """RF branch matches its linear-blend plus renoise algebra."""

    def _const_half(x, sigma, **kwargs):
        return 0.5 * x + 0.1

    sigmas = _rf_sigmas(6)
    start = torch.ones(1, 2, 4, 4) * 0.7
    # eta zero equals the RF drift without renoise.
    got = S.sample_euler_m_RF(_const_half, start.clone(), sigmas.clone(), disable=True, eta=0.0)
    cur = start.clone()
    n = int(cur.shape[0])
    for i in range(len(sigmas) - 1):
        s = float(sigmas[i].item())
        sn = float(sigmas[i + 1].item())
        D = _const_half(cur, sigmas[i] * cur.new_ones([n]))
        if sn == 0.0:
            cur = D.clone()
        else:
            ratio = (sn * (1 + (sn / s - 1) * 0.0)) / s
            cur = ratio * cur + (1 - ratio) * D
    assert bool(torch.allclose(got, cur, atol=1e-6))
    # eta one stays finite and terminal exact for a constant model.
    def _const(x, sigma, **kwargs):
        return torch.ones_like(x) * 0.9

    out = S.sample_euler_m_RF(_const, torch.zeros(1, 1, 2, 2), torch.tensor([0.9, 0.4, 0.0]), disable=True)
    assert bool(torch.allclose(out, torch.ones_like(out) * 0.9))
    for steps in (4, 8):
        sig = _rf_sigmas(steps)
        a = S.sample_euler_m_RF(_const_half, start.clone(), sig.clone(), disable=True, eta=1.0)
        assert bool(torch.isfinite(a).all().item())


def test_dtype_device_batch_preserved():
    """Output keeps dtype and batch shape of the input."""
    sigmas = _karras_like(6)
    start = torch.ones(2, 3, 4, 4, dtype=torch.float32) * 0.5
    out = S.sample_euler_m(_fake_model, start.clone(), sigmas.clone(), disable=True)
    assert out.dtype == start.dtype
    assert out.device == start.device
    assert tuple(out.shape) == tuple(start.shape)
    start64 = torch.ones(1, 2, 4, 4, dtype=torch.float64) * 0.3
    out64 = S.sample_euler_m(_fake_model, start64.clone(), sigmas.clone(), disable=True)
    assert out64.dtype == torch.float64
    # Input is not mutated.
    probe = torch.ones(1, 2, 4, 4)
    snapshot = probe.clone()
    S.sample_euler_m(_fake_model, probe, sigmas.clone(), disable=True)
    assert torch.equal(probe, snapshot)


def test_guards_fail_closed():
    """Non finite outputs, shapes, and bad args fail closed."""

    def _nan(x, sigma, **kwargs):
        return torch.full_like(x, float("nan"))

    def _inf(x, sigma, **kwargs):
        return torch.full_like(x, float("inf"))

    sigmas = _karras_like(4)
    start = torch.ones(1, 2, 4, 4)
    for bad_model in (_nan, _inf):
        try:
            S.sample_euler_m(bad_model, start.clone(), sigmas.clone(), disable=True)
            assert False
        except RuntimeError:
            pass
    bad = sigmas.clone()
    bad[1] = float("nan")
    try:
        S.sample_euler_m(_fake_model, start.clone(), bad, disable=True)
        assert False
    except ValueError:
        pass

    def _wrong_shape(x, sigma, **kwargs):
        return torch.ones(1, 3, 4, 4)

    try:
        S.sample_euler_m(_wrong_shape, start.clone(), sigmas.clone(), disable=True)
        assert False
    except ValueError:
        pass
    try:
        S.sample_euler_m(_fake_model, start.clone(), sigmas.clone(), disable=True, eta=-1.0)
        assert False
    except ValueError:
        pass
    try:
        S.sample_euler_m(_fake_model, start.clone(), sigmas.clone(), disable=True, s_noise=-0.5)
        assert False
    except ValueError:
        pass
    try:
        S.sample_euler_m(_fake_model, start.clone(), sigmas.clone(), disable=True, extra_args=[])
        assert False
    except ValueError:
        pass
