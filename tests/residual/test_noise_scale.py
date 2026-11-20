"""``residual_noise`` (``my_project/wrappers/residual_policy/noise_scale.py``): the NLL fit recovers a known std, the
fitted file round-trips and is refused when it does not fit the run, the MLP scale reads the right input and converts
into r units, and the section's rules."""

from types import SimpleNamespace

import pytest
import torch

from my_project.cfg import ResidualNoiseCfg
from my_project.wrappers.residual_policy import noise_scale as ns
from my_project.wrappers.residual_policy.additive import AdditiveResidual
from my_project.wrappers.residual_policy.multiply import MultiplyResidual


def true_std(x):
    """dim 0: 0.05 + 0.5 |x0|; dim 1: 0.2 everywhere."""
    return torch.stack([0.05 + 0.5 * x[:, 0].abs(), torch.full_like(x[:, 0], 0.2)], dim=-1)


def synthetic(n=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, generator=g) * 2 - 1
    return x, true_std(x) * torch.randn(n, 2, generator=g)


def test_the_nll_fit_recovers_a_known_std_and_beats_a_constant():
    x, c = synthetic()
    out = ns.fit_std_mlp(x, c, hidden=(64, 64), epochs=200, patience=15, lr=3e-3, log_every=0)
    model, report = out["model"], out["report"]
    grid = torch.rand(2000, 3) * 2 - 1
    with torch.no_grad():
        predicted = model(grid)
    rel = (predicted - true_std(grid)).abs() / true_std(grid)
    assert float(rel.mean()) < 0.1
    assert report["val_nll_per_dim"][0] < report["val_nll_constant_per_dim"][0] - 0.05  # the input matters on dim 0
    assert report["val_nll_per_dim"][1] == pytest.approx(report["val_nll_constant_per_dim"][1], abs=0.02)


def test_the_linear_reference_fit_recovers_a1_and_a0():
    g = torch.Generator().manual_seed(1)
    base = torch.rand(30000, 2, generator=g) * 2 - 1
    c = (0.3 * base.abs() + torch.tensor([0.05, 0.1])) * torch.randn(30000, 2, generator=g)
    fit = ns.fit_linear(base, c)
    assert fit["a1"] == pytest.approx([0.3, 0.3], abs=0.03)
    assert fit["a0"] == pytest.approx([0.05, 0.1], abs=0.01)


def test_the_fitted_file_round_trips_and_a_wrong_file_is_refused(tmp_path):
    x, c = synthetic(2000)
    model = ns.fit_std_mlp(x, c, hidden=(16,), epochs=3, log_every=0)["model"]
    path = tmp_path / "std.pt"
    ns.save_std_mlp(path, model, input="residual", residual="additive", fingerprint={"base": "b"}, report={}, source={})
    loaded = ns.load_std_mlp(path)
    torch.testing.assert_close(loaded["model"](x[:50]), model(x[:50]))
    assert (loaded["input"], loaded["residual"], loaded["fingerprint"]) == ("residual", "additive", {"base": "b"})
    torch.save({"format": "other"}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="fit_noise_std"):
        ns.load_std_mlp(tmp_path / "bad.pt")


def additive(width, scale):
    w = object.__new__(AdditiveResidual)
    w.action_dim, w.scale, w.clip, w.table, w.phase_names = width, scale, None, None, []
    w.obs_width, w.device = 3, torch.device("cpu")
    return w


def mlp_cfg(path, input="residual", gain=1.0, warmup_scale=1.0):
    return SimpleNamespace(residual_noise=SimpleNamespace(mode="mlp", warmup_scale=warmup_scale,
                                                          mlp=SimpleNamespace(path=str(path), input=input, gain=gain)))


class Const(ns.StdMLP):
    """An MLP stand-in whose std is the first input column's absolute value, repeated per output dim."""

    def forward(self, x):
        return x[:, :1].abs().expand(-1, self.out_dim) + 0.0


def saved(tmp_path, in_dim, input="residual", residual="additive", out_dim=2, fingerprint=None):
    path = tmp_path / f"{input}_{residual}_{in_dim}_{out_dim}.pt"
    ns.save_std_mlp(path, ns.StdMLP(in_dim, out_dim, (4,)), input=input, residual=residual,
                    fingerprint=fingerprint or {"base": "b"}, report={}, source={})
    return path


def test_the_mlp_scale_reads_its_input_times_the_gain_in_r_units(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "base_fingerprint", lambda cfg, combine: {"base": "b"})
    w = additive(2, torch.tensor([0.5, 0.0]))
    obs = torch.tensor([[0.3, 9.0, 9.0, 1.0, 1.0], [-0.1, 9.0, 9.0, 1.0, 1.0]])
    scale = ns.build_noise_scale(mlp_cfg(saved(tmp_path, 5), gain=2.0, warmup_scale=3.0), w)
    scale.sources[0].model = Const(5, 2, (4,))  # a known std: |obs[:, 0]|
    # s = 2 |obs[:, 0]| in sent units; / 0.5 on dim 0, 0 where the scale is 0
    torch.testing.assert_close(scale(obs, "train"), torch.tensor([[1.2, 0.0], [0.4, 0.0]]))
    torch.testing.assert_close(scale(obs, "warmup"), torch.tensor([[3.6, 0.0], [1.2, 0.0]]))
    # base input: the base policy's observation this step, from its wrapper
    w.base_policy = SimpleNamespace(base_obs=torch.tensor([[0.5, 0.0], [0.25, 0.0]]))
    scale = ns.build_noise_scale(mlp_cfg(saved(tmp_path, 2, input="base"), input="base"), w)
    scale.sources[0].model = Const(2, 2, (4,))
    torch.testing.assert_close(scale(obs, "train"), torch.tensor([[1.0, 0.0], [0.5, 0.0]]))
    w.base_policy.base_obs = torch.zeros(3, 2)  # not the rows acted on: refused
    with pytest.raises(ValueError, match="rows"):
        scale(obs, "train")


def test_a_fitted_file_that_does_not_fit_the_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "base_fingerprint", lambda cfg, combine: {"base": "b"})
    w = additive(2, torch.tensor([0.5, 0.5]))
    with pytest.raises(ValueError, match="input"):
        ns.build_noise_scale(mlp_cfg(saved(tmp_path, 5, input="base")), w)
    with pytest.raises(ValueError, match="multiply residual"):
        ns.build_noise_scale(mlp_cfg(saved(tmp_path, 5, residual="multiply")), w)
    with pytest.raises(ValueError, match="predicts 3 dims"):
        ns.build_noise_scale(mlp_cfg(saved(tmp_path, 5, out_dim=3)), w)
    with pytest.raises(ValueError, match="another base"):
        ns.build_noise_scale(mlp_cfg(saved(tmp_path, 5, fingerprint={"base": "other"})), w)
    scale = ns.build_noise_scale(mlp_cfg(saved(tmp_path, 5)), w)
    with pytest.raises(ValueError, match="takes 5 columns"):
        scale(torch.zeros(2, 4), "train")


def full_cfg(mode="linear", learner="ddpg", warmup="registered", train="registered", base=True):
    noise = lambda scale: SimpleNamespace(scale=scale)
    return SimpleNamespace(trainer=SimpleNamespace(learner=learner), base_policy=SimpleNamespace(enabled=base),
                           ddpg=SimpleNamespace(warmup_noise=noise(warmup), train_noise=noise(train)),
                           sac=SimpleNamespace())


def test_the_sections_rules():
    ResidualNoiseCfg().validate(full_cfg(warmup=0.2, train=0.05))  # none, numbers: today's noise
    ResidualNoiseCfg(mode="linear").validate(full_cfg())
    ResidualNoiseCfg(mode="mlp", mlp=ns_mlp("x.pt")).validate(full_cfg(warmup=0.2))
    bad = [
        (ResidualNoiseCfg(mode="gauss"), full_cfg(), "mode"),
        (ResidualNoiseCfg(), full_cfg(), "set residual_noise.mode"),
        (ResidualNoiseCfg(mode="linear"), full_cfg(warmup=0.2, train=0.05), "unused"),
        (ResidualNoiseCfg(mode="linear"), full_cfg(learner="sac"), "trainer.learner"),
        (ResidualNoiseCfg(mode="linear"), full_cfg(base=False), "base_policy"),
        (ResidualNoiseCfg(mode="linear", warmup_scale=0.0), full_cfg(), "warmup_scale"),
        (ResidualNoiseCfg(mode="linear", linear=ns_linear(float("nan"), 0.0)), full_cfg(), "a1"),
        (ResidualNoiseCfg(mode="linear", linear=ns_linear(0.0, [0.1, True])), full_cfg(), "a0"),
        (ResidualNoiseCfg(mode="mlp"), full_cfg(), "mlp.path"),
        (ResidualNoiseCfg(mode="mlp", mlp=ns_mlp("x.pt", input="critic")), full_cfg(), "mlp.input"),
        (ResidualNoiseCfg(mode="mlp", mlp=ns_mlp("x.pt", gain=0.0)), full_cfg(), "mlp.gain"),
    ]
    for section, cfg, match in bad:
        with pytest.raises(ValueError, match=match):
            section.validate(cfg)


def test_the_linear_scale_is_the_magnitude_of_the_fitted_line():
    s = ns.NoiseScale([ns.LinearSource(0, 2, torch.tensor([-2.0, 0.5]), torch.tensor([1.0, -0.1]))], [0, 0],
                      residual="multiply", warmup_scale=1.0)
    obs = torch.tensor([[0.25, 0.1], [1.0, 1.0]])
    # dim 0: |-2 |b| + 1| -> 0.5 at 0.25, |-1| = 1 at 1; dim 1: |0.5 |b| - 0.1| -> 0.05, 0.4
    torch.testing.assert_close(s(obs, "train"), torch.tensor([[0.5, 0.05], [1.0, 0.4]]))
    ResidualNoiseCfg(mode="linear", linear=ns_linear(-1.0, [0.1, -0.1])).validate(full_cfg())  # negatives allowed


def ns_linear(a1, a0):
    from my_project.cfg import LinearNoiseCfg

    return LinearNoiseCfg(a1=a1, a0=a0)


def ns_mlp(path, input="residual", gain=1.0):
    from my_project.cfg import MlpNoiseCfg

    return MlpNoiseCfg(path=path, input=input, gain=gain)


def test_each_epoch_reports_the_nll_its_gain_over_a_constant_and_the_calibration():
    x, c = synthetic(4000)
    seen = []
    out = ns.fit_std_mlp(x, c, hidden=(32, 32), epochs=6, patience=100, log_every=0,
                         on_epoch=lambda epoch, metrics: seen.append((epoch, metrics)))
    assert [e for e, _ in seen] == list(range(6))
    last = seen[-1][1]
    for key in ("train/nll", "train/grad_norm", "train/lr", "val/nll", "val/nll_constant", "val/nll_gain",
                "val/calibration", "val/pred_std_mean", "val_dims/nll_1", "val_dims/calibration_0"):
        assert isinstance(last[key], float), key
    assert last["val/nll_gain"] == pytest.approx(last["val/nll_constant"] - last["val/nll"], abs=1e-6)
    report = out["report"]
    assert report["best_val_nll"] == pytest.approx(min(m["val/nll"] for _, m in seen))
    assert seen[report["best_epoch"]][1]["val/nll"] == pytest.approx(report["best_val_nll"])
    # the calibration is E[c^2 / s^2]: 1 for the true std
    std, target = true_std(x[:1000]), c[:1000]
    metrics = ns.epoch_metrics(0, 0.0, 0.0, std, target, std.pow(2).mean(0).sqrt(), 1e-3)
    assert metrics["val/calibration"] == pytest.approx(1.0, abs=0.1)


def test_each_dim_takes_its_own_source_linear_in_sent_units_constant_in_r_units():
    # additive, scale [2, 4, 0.5]: the linear dims' s / scale, the constant dims' r as it is (whatever the scale)
    scale = torch.tensor([2.0, 4.0, 0.5])
    a1, a0 = torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.1, 0.2, 0.3])
    sources = [ns.LinearSource(0, 3, a1, a0), ns.ConstantSource(torch.tensor([0.05, 0.05, 0.07]))]
    s = ns.NoiseScale(sources, [0, 1, 1], residual="additive", warmup_scale=2.0, scale=scale)
    obs = torch.tensor([[0.4, -1.0, 2.0], [0.0, 0.0, 0.0]])
    expected = torch.tensor([[(0.5 * 0.4 + 0.1) / 2.0, 0.05, 0.07], [0.1 / 2.0, 0.05, 0.07]])
    torch.testing.assert_close(s(obs, "train"), expected)
    torch.testing.assert_close(s(obs, "warmup"), 2.0 * expected)  # the warm-up's: warmup_scale times every dim's
    torch.testing.assert_close(s.for_route(None)(obs, "train"), expected)  # kept by the rebind
    with pytest.raises(ValueError, match="every source serves"):
        ns.NoiseScale(sources, [0, 0, 0], residual="additive", warmup_scale=1.0, scale=scale)
    # the scaled residual, sent units: the linear dims / |base|, the constant dims as they are
    m = ns.NoiseScale(sources, [1, 0, 1], residual="multiply", warmup_scale=1.0, scaled_units="sent", base_cols=(0, 3))
    torch.testing.assert_close(m(obs[:1], "train"), torch.tensor([[0.05, (0.5 + 0.2) / 1.0, 0.07]]))
    # built from the section: mode one source per dim, each source's settings full width; the other order too
    cfg = SimpleNamespace(residual_noise=SimpleNamespace(
        mode=["linear", "constant", "constant"], warmup_scale=1.0, scaled_units="r",
        linear=SimpleNamespace(a1=[0.5] * 3, a0=[0.1, 0.2, 0.3]), constant=SimpleNamespace(r=[9.0, 0.05, 0.07])))
    rows = torch.cat([torch.zeros(2, 3), obs], dim=1)
    torch.testing.assert_close(ns.build_noise_scale(cfg, additive(3, scale))(rows, "train"), expected)
    cfg.residual_noise.mode = ["constant", "linear", "linear"]
    flipped = torch.tensor([[9.0, (0.5 * 1.0 + 0.2) / 4.0, (0.5 * 2.0 + 0.3) / 0.5], [9.0, 0.2 / 4.0, 0.3 / 0.5]])
    torch.testing.assert_close(ns.build_noise_scale(cfg, additive(3, scale))(rows, "train"), flipped)
    cfg.residual_noise.mode = "constant"  # one name: every dim's
    torch.testing.assert_close(ns.build_noise_scale(cfg, additive(3, scale))(rows, "train"),
                               torch.tensor([[9.0, 0.05, 0.07]] * 2))
    cfg.residual_noise.mode = ["linear", "constant"]
    with pytest.raises(ValueError, match="3 wide"):
        ns.build_noise_scale(cfg, additive(3, scale))
    # the section: a list of known sources, each used source's settings checked, the unused ones not
    ResidualNoiseCfg(mode=["linear"] * 3 + ["constant"] * 3).validate(full_cfg())
    ResidualNoiseCfg(mode=["constant"] * 6, linear=ns_linear(float("nan"), 0.0)).validate(full_cfg())
    for bad, match in ((["linear", "none"], "mode"), (["linear", "fixed"], "mode"), ([], "mode"),
                       (["constant"] * 6, "constant.r")):
        section = ResidualNoiseCfg(mode=bad)
        if match == "constant.r":
            section.constant.r = [0.05, -0.1]
        with pytest.raises(ValueError, match=match):
            section.validate(full_cfg())
