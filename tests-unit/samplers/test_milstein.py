"""Tests for Milstein diagonal sampler. Synthetic data only."""

from __future__ import annotations

import importlib.util
import inspect
import math
import pathlib
import sys
import types

import torch


_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CORE_FILE = _ROOT / "milstein_diffusion.py"
_NODE_FILE = _ROOT / "custom_node" / "__init__.py"
_ADAPTER_FILE = _ROOT / "milstein_comfyui" / "integration.py"
_INSTALLER_FILE = _ROOT / "comfy-milstein.sh"
_README_FILE = _ROOT / "README.md"


def _load_core(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load_core(_CORE_FILE, "milstein_diffusion")


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


def _ancestral_manual(model, x0, sigmas, fixed_noise, eta=1.0, s_noise=1.0, extra_args=None):
    """Independent ancestral Euler update using the sampler helpers.

    Uses the same split and drift helpers as the sampler so the additive
    path (alpha zero) is bit-identical when the Milstein correction is off.
    """
    cur = x0.clone()
    n = int(cur.shape[0])
    args = extra_args or {}
    for i in range(len(sigmas) - 1):
        s = sigmas[i]
        sn = sigmas[i + 1]
        s_in = cur.new_ones([n])
        D = model(cur, s * s_in, **args)
        D = D.to(device=cur.device, dtype=cur.dtype)
        if S.is_terminal_sigma(sn):
            cur = D.clone()
            continue
        sigma_down, sigma_up = S._get_ancestral_step(s, sn, eta=float(eta))
        d = S._to_d(cur, s, D)
        dt = sigma_down - s
        cur = cur + d * dt
        try:
            up_is_zero = float(sigma_up) == 0.0
        except Exception:
            up_is_zero = False
        if not (float(eta) == 0.0 or up_is_zero):
            cur = cur + fixed_noise.to(device=cur.device, dtype=cur.dtype) * float(s_noise) * sigma_up
    return cur


def _additive_step_manual(x, sigma, sigma_next, denoised, noise, eta=1.0, s_noise=1.0):
    """One ancestral Euler step without any Milstein correction."""
    if S.is_terminal_sigma(sigma_next):
        return denoised.clone()
    sigma_down, sigma_up = S._get_ancestral_step(sigma, sigma_next, eta=float(eta))
    d = S._to_d(x, sigma, denoised)
    dt = sigma_down - sigma
    out = x + d * dt
    try:
        up_is_zero = float(sigma_up) == 0.0
    except Exception:
        up_is_zero = False
    if float(eta) == 0.0 or up_is_zero:
        return out
    return out + noise.to(device=x.device, dtype=x.dtype) * float(s_noise) * sigma_up


def test_registration_sampler_only():
    """Milstein name is registered without scheduler entries."""
    mock_samplers = types.ModuleType("comfy.samplers")
    mock_samplers.KSAMPLER_NAMES = ["euler"]
    mock_samplers.SAMPLER_NAMES = ["euler"]
    mock_sampling = types.ModuleType("comfy.k_diffusion.sampling")
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.samplers"] = mock_samplers
    sys.modules["comfy.k_diffusion"] = types.ModuleType("comfy.k_diffusion")
    sys.modules["comfy.k_diffusion.sampling"] = mock_sampling
    try:
        spec = importlib.util.spec_from_file_location("milstein_node", str(_NODE_FILE))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for key in ("comfy", "comfy.samplers", "comfy.k_diffusion", "comfy.k_diffusion.sampling"):
            sys.modules.pop(key, None)
    assert "milstein" in mock_samplers.KSAMPLER_NAMES
    assert "milstein" in mock_samplers.SAMPLER_NAMES
    assert getattr(mock_sampling, "sample_milstein") is not None
    assert not hasattr(mock_samplers, "SCHEDULER_NAMES") or "milstein" not in getattr(
        mock_samplers, "SCHEDULER_NAMES", []
    )
    assert not hasattr(mock_samplers, "SCHEDULER_HANDLERS") or "milstein" not in getattr(
        mock_samplers, "SCHEDULER_HANDLERS", {}
    )
    assert mod.NODE_CLASS_MAPPINGS == {}
    assert mod.NODE_DISPLAY_NAME_MAPPINGS == {}
    assert set(mock_samplers.KSAMPLER_NAMES) == {"euler", "milstein"}
    assert set(mock_samplers.SAMPLER_NAMES) == {"euler", "milstein"}
    assert set(k for k in dir(mock_sampling) if k.startswith("sample_")) == {"sample_milstein"}
    src = _NODE_FILE.read_text()
    assert "milstein" in src
    assert src.count("_register(") == 2


def test_signature_defaults():
    """Sampler exposes Comfy signature plus eta, s_noise, noise, alpha, k."""
    for fn_name in ("sample_milstein", "sample_milstein_RF"):
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
            "alpha",
            "k",
        ]
        assert float(sig.parameters["eta"].default) == 1.0
        assert float(sig.parameters["s_noise"].default) == 1.0
        assert sig.parameters["noise_sampler"].default is None
        assert float(sig.parameters["alpha"].default) == 0.15
        assert float(sig.parameters["k"].default) == 1.0
    sig = inspect.signature(S.milstein_step)
    assert list(sig.parameters.keys()) == [
        "x",
        "sigma",
        "sigma_next",
        "denoised",
        "noise",
        "eta",
        "s_noise",
        "alpha",
        "k",
    ]
    src = _CORE_FILE.read_text()
    assert getattr(S.sample_milstein, "__wrapped__", None) is not None or "no_grad" in src


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
    assert "sample_milstein" in text
    assert "sample_milstein_RF" in text
    assert "milstein_diffusion" in text


def test_validate_sigmas_accepts_good():
    """Good schedules pass with tolerant terminal handling."""
    for steps in (2, 10, 20):
        sigmas = _karras_like(steps)
        assert S.validate_sigmas(sigmas) == steps
        assert S.validate_sigmas(sigmas, steps) == steps
        assert S.validate_sigmas(_linear_sigmas(steps)) == steps


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
        S.sample_milstein(_counting, start.clone(), sigmas.clone(), disable=True, eta=0.0)
        assert calls["n"] == steps
        assert calls["n"] == S.count_nfe(steps)


def test_alpha_zero_collapses_to_ancestral():
    """With alpha zero Milstein is bit-identical to the ancestral update."""
    for sigmas in (_karras_like(10), _linear_sigmas(10)):
        start = torch.randn(1, 2, 4, 4, generator=torch.Generator().manual_seed(11))
        fixed = torch.randn(1, 2, 4, 4, generator=torch.Generator().manual_seed(12)) * 0.7

        def _fixed(a, b):
            return fixed.clone()

        got = S.sample_milstein(
            _fake_model,
            start.clone(),
            sigmas.clone(),
            disable=True,
            eta=1.0,
            s_noise=1.0,
            noise_sampler=_fixed,
            alpha=0.0,
        )
        ref = _ancestral_manual(
            _fake_model,
            start.clone(),
            sigmas.clone(),
            fixed.clone(),
            eta=1.0,
            s_noise=1.0,
        )
        assert torch.equal(got, ref)
    # Step helper also collapses bit-identically.
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4)
    D = torch.randn(1, 2, 4, 4)
    n = torch.randn(1, 2, 4, 4)
    a = S.milstein_step(x, 5.0, 3.0, D, n, eta=1.0, s_noise=1.0, alpha=0.0)
    b = _additive_step_manual(x, 5.0, 3.0, D, n, eta=1.0, s_noise=1.0)
    assert torch.equal(a, b)


def test_alpha_nonzero_differs_but_bounded():
    """Nonzero alpha moves the trajectory while staying bounded."""
    sigmas = _karras_like(6)
    start = torch.ones(1, 2, 4, 4) * 0.6
    fixed = torch.ones(1, 2, 4, 4) * 0.4

    def _fixed(a, b):
        return fixed.clone()

    base = S.sample_milstein(
        _fake_model, start.clone(), sigmas.clone(), disable=True, noise_sampler=_fixed, alpha=0.0
    )
    mod = S.sample_milstein(
        _fake_model, start.clone(), sigmas.clone(), disable=True, noise_sampler=_fixed, alpha=0.15
    )
    assert not torch.equal(base, mod)
    assert bool(torch.isfinite(mod).all().item())
    # Scale stays inside [1-alpha, 1+alpha].
    probe = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    m = S.milstein_scale(probe, alpha=0.15, k=1.0)
    assert bool(((m >= 1.0 - 0.15 - 1e-6) & (m <= 1.0 + 0.15 + 1e-6)).all().item())
    assert bool(torch.allclose(S.milstein_scale(probe, alpha=0.0), torch.ones_like(probe)))
    # Finite-difference derivative matches the analytic tanh slope.
    x = torch.tensor([0.0, 0.5, -0.5])
    fd = S.milstein_scale_derivative(x, alpha=0.15, k=1.0)
    analytic = 0.15 * (1.0 - torch.tanh(x / 1.0) ** 2) / 1.0
    assert bool(torch.allclose(fd, analytic, atol=2e-3))


def test_gbm_strong_order():
    """Milstein beats Euler-Maruyama on a GBM toy (strong order)."""
    torch.manual_seed(7)
    mu = 0.1
    beta = 0.3
    x0 = 1.0
    dt = 0.05
    n_paths = 2000
    dW = torch.randn(n_paths) * (dt**0.5)
    exact = x0 * torch.exp((mu - 0.5 * beta**2) * dt + beta * dW)
    euler = x0 + mu * x0 * dt + beta * x0 * dW
    b = beta * x0
    bp = beta
    mil = x0 + mu * x0 * dt + beta * x0 * dW + S.milstein_correction(
        torch.full_like(dW, b), torch.full_like(dW, bp), dW, dt
    )
    mse_euler = float(((euler - exact) ** 2).mean().item())
    mse_mil = float(((mil - exact) ** 2).mean().item())
    assert mse_mil < mse_euler
    assert mse_euler / max(mse_mil, 1e-12) > 5.0
    # CIR specialization: 0.5*b*b' collapses to xi^2/4 independent of v.
    xi = 0.5
    v = torch.tensor([1.0, 2.0])
    dWt = torch.tensor([0.7, -0.4])
    dtt = 0.1
    b_cir = xi * torch.sqrt(v)
    bp_cir = xi / (2.0 * torch.sqrt(v))
    got = S.milstein_correction(b_cir, bp_cir, dWt, dtt)
    ref = 0.25 * xi**2 * (dWt * dWt - dtt)
    assert bool(torch.allclose(got, ref, atol=1e-6))


def test_eta_zero_equals_euler():
    """With eta zero the SDE collapses exactly to the Euler ODE."""
    for sigmas in (_karras_like(10), _linear_sigmas(10)):
        start = torch.ones(1, 2, 4, 4) * 0.7
        got = S.sample_milstein(
            _fake_model,
            start.clone(),
            sigmas.clone(),
            disable=True,
            eta=0.0,
            noise_sampler=lambda a, b: torch.zeros_like(start),
            alpha=0.15,
        )
        ref = _euler_run(_fake_model, start.clone(), sigmas.clone())
        assert bool(torch.allclose(got, ref, atol=1e-6))
    # eta zero needs no noise even with modulation on.
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 4)
    D = torch.randn(1, 2, 4, 4)
    nxt = S.milstein_step(x, 5.0, 3.0, D, None, eta=0.0, alpha=0.15)
    rho = 3.0 / 5.0
    assert bool(torch.allclose(nxt, rho * x + (1.0 - rho) * D))


def test_determinism_and_seed_sensitivity():
    """Fixed sampler is deterministic, default sampler reacts to seed."""
    sigmas = _karras_like(6)
    start = torch.ones(1, 2, 4, 4) * 0.6
    fixed = lambda a, b: torch.ones(1, 2, 4, 4) * 0.25  # noqa: E731
    first = S.sample_milstein(
        _fake_model, start.clone(), sigmas.clone(), disable=True, noise_sampler=fixed, alpha=0.15
    )
    second = S.sample_milstein(
        _fake_model, start.clone(), sigmas.clone(), disable=True, noise_sampler=fixed, alpha=0.15
    )
    assert torch.equal(first, second)
    seeded_a = S.sample_milstein(
        _fake_model, start.clone(), sigmas.clone(), disable=True, extra_args={"seed": 7}
    )
    seeded_b = S.sample_milstein(
        _fake_model, start.clone(), sigmas.clone(), disable=True, extra_args={"seed": 7}
    )
    seeded_c = S.sample_milstein(
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
        S.sample_milstein(_fake_model, torch.ones(1, 2, 4, 4), sigmas.clone(), callback=_cb, disable=True)
        assert seen == list(range(steps))
        for data in payloads:
            assert set(("x", "i", "sigma", "sigma_hat", "denoised")) <= set(data.keys())


def test_terminal_zero_returns_denoised():
    """Terminal step returns denoised prediction directly."""

    def _const(x, sigma, **kwargs):
        return torch.ones_like(x) * 1.5

    sigmas = torch.tensor([3.0, 1.0, 0.0])
    out = S.sample_milstein(_const, torch.zeros(1, 1, 2, 2), sigmas, disable=True)
    assert bool(torch.allclose(out, torch.ones_like(out) * 1.5))
    near = torch.tensor([3.0, 1.0, 1e-12])
    out_near = S.sample_milstein(_const, torch.zeros(1, 1, 2, 2), near, disable=True)
    assert bool(torch.allclose(out_near, torch.ones_like(out_near) * 1.5))
    D = torch.ones(1, 2, 3, 3) * 2.0
    x = torch.ones(1, 2, 3, 3) * 5.0
    assert torch.equal(S.milstein_step(x, 0.5, 0.0, D, None, eta=1.0, alpha=0.15), D)


def test_rf_raises_fail_closed():
    """Rectified-flow twin is deferred and fails closed."""
    try:
        S.sample_milstein_RF(_fake_model, torch.ones(1, 1, 2, 2), torch.tensor([0.9, 0.4, 0.0]))
        assert False
    except NotImplementedError:
        pass
    # CONST models dispatch to the raising stub instead of sampling.

    class _Inner:
        pass

    class _Model:
        pass

    try:
        from comfy.model_sampling import CONST as _Const  # noqa: F401
        has_const = True
    except Exception:
        has_const = False
    if has_const:
        from comfy.model_sampling import CONST as _Const

        sampling = _Const()
        inner2 = types.SimpleNamespace(model_sampling=sampling)
        inner1 = types.SimpleNamespace(inner_model=inner2)
        model = types.SimpleNamespace(inner_model=inner1)
        try:
            S.sample_milstein(_fake_model if False else model, torch.ones(1, 1, 2, 2), torch.tensor([0.9, 0.4, 0.0]))
            assert False
        except NotImplementedError:
            pass
    else:
        # Without Comfy CONST, the stub still raises when called directly.
        assert True


def test_dtype_device_batch_preserved():
    """Output keeps dtype and batch shape of the input."""
    sigmas = _karras_like(6)
    start = torch.ones(2, 3, 4, 4, dtype=torch.float32) * 0.5
    out = S.sample_milstein(_fake_model, start.clone(), sigmas.clone(), disable=True)
    assert out.dtype == start.dtype
    assert out.device == start.device
    assert tuple(out.shape) == tuple(start.shape)
    start64 = torch.ones(1, 2, 4, 4, dtype=torch.float64) * 0.3
    out64 = S.sample_milstein(_fake_model, start64.clone(), sigmas.clone(), disable=True)
    assert out64.dtype == torch.float64
    probe = torch.ones(1, 2, 4, 4)
    snapshot = probe.clone()
    S.sample_milstein(_fake_model, probe, sigmas.clone(), disable=True)
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
            S.sample_milstein(bad_model, start.clone(), sigmas.clone(), disable=True)
            assert False
        except RuntimeError:
            pass
    bad = sigmas.clone()
    bad[1] = float("nan")
    try:
        S.sample_milstein(_fake_model, start.clone(), bad, disable=True)
        assert False
    except ValueError:
        pass

    def _wrong_shape(x, sigma, **kwargs):
        return torch.ones(1, 3, 4, 4)

    try:
        S.sample_milstein(_wrong_shape, start.clone(), sigmas.clone(), disable=True)
        assert False
    except ValueError:
        pass
    try:
        S.sample_milstein(_fake_model, start.clone(), sigmas.clone(), disable=True, eta=-1.0)
        assert False
    except ValueError:
        pass
    try:
        S.sample_milstein(_fake_model, start.clone(), sigmas.clone(), disable=True, s_noise=-0.5)
        assert False
    except ValueError:
        pass
    try:
        S.sample_milstein(_fake_model, start.clone(), sigmas.clone(), disable=True, extra_args=[])
        assert False
    except ValueError:
        pass
    for bad_alpha in (-0.1, 1.0, float("nan")):
        try:
            S.sample_milstein(_fake_model, start.clone(), sigmas.clone(), disable=True, alpha=bad_alpha)
            assert False
        except ValueError:
            pass
    for bad_k in (0.0, -1.0, float("inf")):
        try:
            S.sample_milstein(_fake_model, start.clone(), sigmas.clone(), disable=True, k=bad_k)
            assert False
        except ValueError:
            pass


def test_branch_layout_exclusive():
    """Branch holds exactly one sampler core, adapter, installer, and test."""
    assert _CORE_FILE.exists()
    assert _ADAPTER_FILE.exists()
    assert (_ROOT / "milstein_comfyui" / "__init__.py").exists()
    assert _INSTALLER_FILE.exists()
    assert _NODE_FILE.exists()
    top_py = sorted(p.name for p in _ROOT.glob("*.py"))
    assert top_py == ["milstein_diffusion.py"]
    assert not any("draft" in n.lower() or "genie" in n.lower() for n in top_py)
    installers = sorted(p.name for p in _ROOT.glob("comfy-*.sh"))
    assert installers == ["comfy-milstein.sh"]
    sampler_tests = sorted(p.name for p in (_ROOT / "tests-unit" / "samplers").glob("test_*.py"))
    assert sampler_tests == ["test_milstein.py"]
    adapter_dirs = sorted(p.name for p in _ROOT.glob("*_comfyui") if p.is_dir())
    assert adapter_dirs == ["milstein_comfyui"]


def test_installer_and_readme_presence():
    """Installer script and README section exist for the sampler with SDE note."""
    assert _INSTALLER_FILE.exists()
    text = _INSTALLER_FILE.read_text()
    assert "milstein" in text.lower()
    assert "milstein-diffusion" in text
    assert "custom_nodes" in text
    assert _README_FILE.exists()
    readme = _README_FILE.read_text()
    assert "Milstein" in readme
    assert "milstein" in readme.lower()
    assert "SDE" in readme
    assert "eta" in readme.lower()
    assert "sampler/milstein-diagonal" in readme
    assert "comfy-milstein.sh" in readme
