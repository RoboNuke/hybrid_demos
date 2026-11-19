"""``residual_noise.*``: DDPG's noise scale per env and dim (the package's ``register_noise_scale``, for a learner phase
with ``ddpg.<phase>_noise.scale: registered``), from the base action or the observation.

Every mode computes ``s`` per env and action dim:

* ``linear``: ``s = |a1 * |base| + a0|``, ``base`` the base action this step (read from the agent's observation, where it
  sits right after the observation columns); the magnitude, so a fitted line with a negative slope (or intercept) never
  gives a negative scale, and the line is otherwise used as fitted;
* ``mlp``: an MLP fitted to the demo corrections by Gaussian NLL (``scripts/fit_noise_std.py``, :func:`fit_std_mlp`):
  ``s(x)^2`` estimates ``E[c^2 | x]``, ``c`` the demo's correction on the base, times ``mlp.gain``; ``x`` the agent's
  observation (``mlp.input: residual``) or the observation the base policy is given (``base``).

``s`` is in the residual's own correction units, and turned into r units -- what the package's noise adds to the actor's
action -- by the residual's rule:

* additive (``sent = base + scale * r``): ``s`` is in the sent action's units, so ``r``'s scale is ``s / scale`` (0 on a
  dim whose scale is 0); with per-phase scales, each routed learner's own phase's (:meth:`NoiseScale.for_route`);
* scaled (``sent = r * base``), ``scaled_units: r``: ``s`` is already in r units (the multiplier's), used as it is;
  ``r * base`` multiplies the noise by the base;
* scaled, ``scaled_units: sent``: ``s`` is in the sent action's units and the noise is added to ``r * base``: r's scale
  is ``s / |base|`` (``|base|`` floored at :data:`SENT_FLOOR`), so ``(r_det + e / base) * base = r_det * base + e`` -- the
  stored r carries ``e / base`` and the critics' ``r * base`` sees exactly the sent action. Under the floor the noise
  shrinks (``r * base`` can add nothing where the base is 0).

In the warm-up phase the scale is ``warmup_scale * s``. A gaussian phase draws ``N(0, scale)``, a uniform one
``U(-scale, scale)`` (the package's ``distribution``). Device ops only: it runs every acting step.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

__all__ = ["NoiseScale", "LinearScale", "MlpScale", "StdMLP", "fit_std_mlp", "save_std_mlp", "load_std_mlp",
           "build_noise_scale", "nll", "MLP_FORMAT", "check_raw_fingerprint", "base_fingerprint"]

#: the fitted MLP file's format tag (``scripts/fit_noise_std.py``)
MLP_FORMAT = "residual_noise_std_mlp/v1"
#: ``scaled_units: sent``: the smallest ``|base|`` the noise is divided by
SENT_FLOOR = 1e-6


# ---------------------------------------------------------------------------------------------------------- the MLP
class StdMLP(nn.Module):
    """``x -> s(x)``: inputs standardized by the fit data's mean and std (clipped at +-``input_clip``), a ReLU MLP, a
    softplus head plus ``floor`` so the std stays positive."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 256), floor: float = 1e-4,
                 input_clip: float = 10.0) -> None:
        super().__init__()
        self.in_dim, self.out_dim, self.hidden = int(in_dim), int(out_dim), [int(h) for h in hidden]
        self.floor, self.input_clip = float(floor), float(input_clip)
        layers: List[nn.Module] = []
        width = self.in_dim
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, self.out_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("in_mean", torch.zeros(self.in_dim))
        self.register_buffer("in_std", torch.ones(self.in_dim))

    def set_normalizer(self, inputs: torch.Tensor, std_floor: float = 1e-3) -> None:
        self.in_mean.copy_(inputs.mean(dim=0))
        self.in_std.copy_(inputs.std(dim=0).clamp_min(std_floor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = ((x - self.in_mean) / self.in_std).clamp(-self.input_clip, self.input_clip)
        return nn.functional.softplus(self.net(z)) + self.floor


def nll(std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The zero-mean Gaussian negative log-likelihood per element, without the constant: ``log s + c^2 / (2 s^2)``.
    Its minimizer over ``s`` is ``s^2 = E[c^2]``."""
    return torch.log(std) + 0.5 * (target / std) ** 2


def fit_std_mlp(inputs: torch.Tensor, targets: torch.Tensor, *, hidden: Sequence[int] = (256, 256),
                floor: float = 1e-4, epochs: int = 500, lr: float = 1e-3, batch_size: int = 1024,
                weight_decay: float = 0.0, val_fraction: float = 0.1, patience: int = 20, seed: int = 0,
                device: Any = "cpu", log_every: int = 10, on_epoch: Any = None) -> Dict[str, Any]:
    """Fit :class:`StdMLP` on ``inputs`` ``(N, in)`` -> the std of ``targets`` ``(N, A)`` by the zero-mean Gaussian NLL
    (Adam, a random ``val_fraction`` held out, the best validation epoch kept, stop after ``patience`` epochs without
    improvement). Returns ``{model, report}``; the report has the per-dim validation NLL of the MLP and of the best
    constant std (the targets' RMS on the training rows) for comparison. ``on_epoch(epoch, metrics)``, when given, gets
    each epoch's :func:`epoch_metrics` (floats, wandb-style names)."""
    if inputs.shape[0] != targets.shape[0] or inputs.dim() != 2 or targets.dim() != 2:
        raise ValueError(f"inputs {tuple(inputs.shape)} and targets {tuple(targets.shape)}: two (N, ...) matrices")
    g = torch.Generator().manual_seed(int(seed))
    n = inputs.shape[0]
    order = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * float(val_fraction))))
    val, train = order[:n_val], order[n_val:]
    x_tr, y_tr = inputs[train].to(device), targets[train].to(device)
    x_va, y_va = inputs[val].to(device), targets[val].to(device)
    torch.manual_seed(int(seed))
    model = StdMLP(inputs.shape[1], targets.shape[1], hidden, floor).to(device)
    model.set_normalizer(x_tr)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    constant = y_tr.pow(2).mean(dim=0).sqrt().clamp_min(floor)
    best, best_state, best_epoch, stale, history = math.inf, None, 0, 0, []
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(x_tr.shape[0], generator=g).to(device)
        total, grad_total, batches = 0.0, 0.0, 0
        for start in range(0, x_tr.shape[0], int(batch_size)):
            idx = perm[start:start + int(batch_size)]
            loss = nll(model(x_tr[idx]), y_tr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            grad = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
            opt.step()
            total, grad_total, batches = total + float(loss), grad_total + float(grad), batches + 1
        model.eval()
        with torch.no_grad():
            std_va = model(x_va)
            val_loss = float(nll(std_va, y_va).mean())
        history.append(val_loss)
        if on_epoch is not None:
            on_epoch(epoch, epoch_metrics(epoch, total / batches, grad_total / batches, std_va, y_va, constant, lr))
        if val_loss < best - 1e-6:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if log_every and (epoch % int(log_every) == 0):
            print(f"[fit_noise_std] epoch {epoch}: validation NLL {val_loss:.5f} (best {best:.5f})", flush=True)
        if stale >= int(patience):
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        std_va = model(x_va)
        report = {
            "epochs": len(history),
            "best_epoch": best_epoch,
            "best_val_nll": best,
            "val_nll_per_dim": nll(std_va, y_va).mean(dim=0).tolist(),
            "val_nll_constant_per_dim": nll(constant.expand_as(y_va), y_va).mean(dim=0).tolist(),
            "target_rms_per_dim": y_va.pow(2).mean(dim=0).sqrt().tolist(),
            "predicted_std_mean_per_dim": std_va.mean(dim=0).tolist(),
            "predicted_rms_std_per_dim": std_va.pow(2).mean(dim=0).sqrt().tolist(),
            "rows": {"train": int(train.numel()), "val": int(val.numel())},
        }
    return {"model": model.cpu(), "report": report}


def epoch_metrics(epoch: int, train_nll: float, grad_norm: float, std: torch.Tensor, target: torch.Tensor,
                  constant: torch.Tensor, lr: float) -> Dict[str, float]:
    """One epoch's metrics on the validation rows (``std`` the MLP's prediction, ``target`` the corrections,
    ``constant`` the best constant std): the NLL overall and per dim, the gain over the constant std (> 0: the input
    helps), the calibration ``E[c^2 / s^2]`` (1 when the predicted variance matches), and the predicted std's mean."""
    per_dim = nll(std, target).mean(dim=0)
    const = nll(constant.expand_as(target), target).mean(dim=0)
    z2 = (target / std).pow(2).mean(dim=0)
    out = {"epoch": float(epoch), "train/nll": train_nll, "train/grad_norm": grad_norm, "train/lr": float(lr),
           "val/nll": float(per_dim.mean()), "val/nll_constant": float(const.mean()),
           "val/nll_gain": float((const - per_dim).mean()), "val/calibration": float(z2.mean()),
           "val/pred_std_mean": float(std.mean())}
    for d in range(per_dim.shape[0]):
        out[f"val_dims/nll_{d}"] = float(per_dim[d])
        out[f"val_dims/nll_gain_{d}"] = float(const[d] - per_dim[d])
        out[f"val_dims/calibration_{d}"] = float(z2[d])
        out[f"val_dims/pred_std_mean_{d}"] = float(std[:, d].mean())
    return out


def slice_linear_fit(base: Any, residual: Any, *, num_bins: int = 50, x_quantile: float = 0.995,
                     min_points: int = 5) -> Dict[str, List[float]]:
    """``data_analysis/demo_residual_analysis.ipynb``'s last cell, per dim (X_MODE "magnitude", STAT "std", FIT with
    degree 1, FIT_WEIGHTED, FIT_TRIM_BINS (0, 0)): ``num_bins`` equal slices of ``|base|`` over ``[0, its x_quantile
    quantile]``, the residual's std (population, numpy's) in every slice with ``min_points`` points or more, and the
    least-squares line through the slices' (centre, std), each slice weighted by its count: ``std ~ a1 * |base| + a0``.
    In the arrays' own units (the notebook's plot scale changes ``a0`` only, by the scale). Returns ``{a1, a0, r2,
    slices}`` lists, one entry per dim."""
    import numpy as np

    b, y = np.abs(np.asarray(base, dtype=np.float64)), np.asarray(residual, dtype=np.float64)
    out: Dict[str, List[float]] = {"a1": [], "a0": [], "r2": [], "slices": []}
    for d in range(b.shape[1]):
        x, r = b[:, d], y[:, d]
        lo, hi = 0.0, float(np.quantile(x, x_quantile))
        edges = np.linspace(lo, hi, num_bins + 1)
        if hi <= lo:  # a constant base action: one slice holds it all
            edges = np.array([lo - 0.5, lo + 0.5])
        slice_of = np.digitize(x, edges[1:-1])
        inside = (x >= edges[0]) & (x <= edges[-1])
        centres, values, counts = [], [], []
        for k in range(len(edges) - 1):
            ys = r[inside & (slice_of == k)]
            if len(ys) >= min_points:
                centres.append((edges[k] + edges[k + 1]) / 2)
                values.append(ys.std())
                counts.append(len(ys))
        cx, cy, w = np.asarray(centres), np.asarray(values), np.asarray(counts, dtype=np.float64)
        if len(cx) <= 2:
            out["a1"].append(float("nan")), out["a0"].append(float("nan"))
            out["r2"].append(float("nan")), out["slices"].append(len(cx))
            continue
        sw = np.sqrt(w)
        solved, *_ = np.linalg.lstsq(np.stack([cx, np.ones_like(cx)], axis=1) * sw[:, None], cy * sw, rcond=None)
        mean = np.average(cy, weights=w)
        ss_res = np.sum(w * (cy - np.polyval(solved, cx)) ** 2)
        ss_tot = np.sum(w * (cy - mean) ** 2)
        out["a1"].append(float(solved[0]))
        out["a0"].append(float(solved[1]))
        out["r2"].append(float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))
        out["slices"].append(len(cx))
    return out


def fit_linear(base: torch.Tensor, targets: torch.Tensor, *, steps: int = 3000, lr: float = 1e-2,
               device: Any = "cpu") -> Dict[str, List[float]]:
    """A reference for ``mode: linear``: per dim the ``(a1, a0) >= 0`` minimizing the same NLL with
    ``s = a1 * |base| + a0`` (Adam on softplus parameters)."""
    b, y = base.abs().to(device), targets.to(device)
    rms = y.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
    inv = lambda v: torch.log(torch.expm1(v))  # softplus^-1
    p1 = inv(torch.full_like(rms, 1e-3)).requires_grad_(True)
    p0 = inv(rms.clone()).requires_grad_(True)
    opt = torch.optim.Adam([p1, p0], lr=lr)
    for _ in range(int(steps)):
        a1, a0 = nn.functional.softplus(p1), nn.functional.softplus(p0)
        loss = nll(a1 * b + a0 + 1e-6, y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        a1, a0 = nn.functional.softplus(p1), nn.functional.softplus(p0)
        return {"a1": a1.cpu().tolist(), "a0": a0.cpu().tolist(),
                "nll_per_dim": nll(a1 * b + a0 + 1e-6, y).mean(dim=0).cpu().tolist()}


def save_std_mlp(path: Any, model: StdMLP, *, input: str, residual: str, fingerprint: Dict[str, Any],
                 report: Dict[str, Any], source: Dict[str, Any]) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": MLP_FORMAT,
        "arch": {"in_dim": model.in_dim, "out_dim": model.out_dim, "hidden": model.hidden, "floor": model.floor,
                 "input_clip": model.input_clip},
        "state_dict": model.state_dict(),
        "input": input,
        "residual": residual,
        "fingerprint": fingerprint,
        "report": report,
        "source": source,
    }, path)


def load_std_mlp(path: Any) -> Dict[str, Any]:
    """The fitted file as ``{model, input, residual, fingerprint, report, source}``; refused when it is not one."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or data.get("format") != MLP_FORMAT:
        raise ValueError(f"residual_noise.mlp.path {path} is not a fitted noise-std MLP ({MLP_FORMAT}): fit one with "
                         "scripts/fit_noise_std.py")
    arch = data["arch"]
    model = StdMLP(arch["in_dim"], arch["out_dim"], arch["hidden"], arch["floor"], arch["input_clip"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    return {**{k: data[k] for k in ("input", "residual", "fingerprint", "report", "source")}, "model": model}


# ---------------------------------------------------------------------------------------------------- fingerprints
def base_fingerprint(cfg: Any, combine: Any) -> Dict[str, Any]:
    """This run's base and task, as the raw demo files record them (:func:`~my_project.raw_demos.raw_fingerprint`; the
    dataset entry left out: it names the demos, not the base)."""
    from my_project.raw_demos import raw_fingerprint

    now = raw_fingerprint(cfg.base_policy, combine.base_policy.policy, "", cfg.task_name, getattr(cfg.wandb, "entity", None))
    return {k: v for k, v in now.items() if k != "dataset"}


def check_raw_fingerprint(made: Dict[str, Any], now: Dict[str, Any], what: str, fix: str) -> None:
    off = {k: (made.get(k), v) for k, v in now.items() if made.get(k) != v}
    if off:
        raise ValueError(f"{what} was made for another base or task (file -> this run): {off}; {fix}")


# ------------------------------------------------------------------------------------------------------ the scales
class NoiseScale:
    """The common part: the conversion of ``s`` into r units (additive: ``/ scale``, per routed phase with per-phase
    scales; scaled: as it is) and the warm-up factor. Subclasses implement :meth:`s`."""

    def __init__(self, *, residual: str, warmup_scale: float, scale: Optional[torch.Tensor] = None,
                 table: Optional[torch.Tensor] = None, names: Sequence[str] = (), route: Optional[str] = None,
                 scaled_units: str = "r", base_cols: Optional[Sequence[int]] = None) -> None:
        if residual not in ("additive", "multiply"):
            raise ValueError(f"residual must be additive or multiply, got {residual!r}")
        if scaled_units == "sent" and (residual != "multiply" or base_cols is None):
            raise ValueError("scaled_units: sent is the scaled residual's, and reads the base action's columns")
        self.residual, self.warmup_scale = residual, float(warmup_scale)
        self.scaled_units, self.base_cols = scaled_units, None if base_cols is None else (int(base_cols[0]), int(base_cols[1]))
        self.table, self.names, self.route = table, list(names), route
        if table is not None and route is not None:
            scale = table[self.names.index(route)]
        self.scale = scale

    def _rebind(self, route: Optional[str]) -> "NoiseScale":
        raise NotImplementedError

    def for_route(self, route: Optional[str]) -> "NoiseScale":
        """This scale for the learner of phase ``route`` (its own phase's additive scale; None unrouted)."""
        return self._rebind(route)

    def s(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, observations: torch.Tensor, phase: str) -> torch.Tensor:
        s = self.s(observations)
        if self.residual == "additive":
            if self.scale is None:
                raise NotImplementedError("the residual noise with a per-phase additive scale needs one residual per phase "
                                          "(route_labels): one learner's observations do not say which phase's scale a "
                                          "row has")
            on = self.scale > 0
            s = torch.where(on, s / torch.where(on, self.scale, torch.ones_like(self.scale)), torch.zeros_like(s))
        elif self.scaled_units == "sent":  # the noise added to r * base: r takes s / |base|
            base = observations[..., self.base_cols[0]:self.base_cols[1]].abs().clamp_min(SENT_FLOOR)
            s = s / base
        if phase == "warmup":
            s = s * self.warmup_scale
        return s

    def _common(self) -> Dict[str, Any]:
        return {"residual": self.residual, "warmup_scale": self.warmup_scale,
                "scale": self.scale if self.table is None else None, "table": self.table, "names": self.names,
                "scaled_units": self.scaled_units, "base_cols": self.base_cols}


class LinearScale(NoiseScale):
    """``s = |a1 * |base| + a0|`` per dim, ``base`` at ``[start, stop)`` of the agent's observation."""

    def __init__(self, start: int, stop: int, a1: torch.Tensor, a0: torch.Tensor, **common) -> None:
        super().__init__(**common)
        self.start, self.stop, self.a1, self.a0 = int(start), int(stop), a1, a0

    def _rebind(self, route):
        return LinearScale(self.start, self.stop, self.a1, self.a0, **self._common(), route=route)

    def s(self, observations: torch.Tensor) -> torch.Tensor:
        return (self.a1 * observations[..., self.start:self.stop].abs() + self.a0).abs()


class MlpScale(NoiseScale):
    """``s = gain * mlp(x)``, ``x`` the agent's observation (``input: residual``) or the observation the base policy was
    given this step (``input: base``: :attr:`BasePolicy.base_obs <my_project.wrappers.residual_policy.base_policy.
    BasePolicy>`, read from the wrapper, so it serves the live env's acting step only)."""

    def __init__(self, model: StdMLP, input: str, gain: float, base_wrapper: Any = None, **common) -> None:
        super().__init__(**common)
        self.model, self.input, self.gain, self.base_wrapper = model, input, float(gain), base_wrapper
        if input == "base" and base_wrapper is None:
            raise ValueError("residual_noise.mlp.input: base reads the base policy's observation, and no base policy wrapper was given")

    def _rebind(self, route):
        return MlpScale(self.model, self.input, self.gain, self.base_wrapper, **self._common(), route=route)

    def s(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations if self.input == "residual" else self.base_wrapper.base_obs
        if x.shape[0] != observations.shape[0] or x.shape[-1] != self.model.in_dim:
            raise ValueError(f"residual_noise.mlp ({self.input} input): the MLP takes {self.model.in_dim} columns, got "
                             f"{tuple(x.shape)} for {observations.shape[0]} rows")
        with torch.no_grad():
            return self.gain * self.model(x)


def _per_dim(value: Any, width: int, what: str, device: Any) -> torch.Tensor:
    numbers = [float(value)] if isinstance(value, (int, float)) else [float(v) for v in value]
    if len(numbers) not in (1, width):
        raise ValueError(f"{what} has {len(numbers)} numbers; the action is {width} wide")
    return torch.tensor(numbers * (width if len(numbers) == 1 else 1), dtype=torch.float32, device=device)


def build_noise_scale(cfg: Any, combine: Any) -> Optional[NoiseScale]:
    """The ``register_noise_scale`` answer for this run's residual ``combine``: None with ``residual_noise.mode: none``."""
    rn = cfg.residual_noise
    if rn.mode == "none":
        return None
    residual = combine.noise_kind
    width, device = combine.action_dim, combine.device
    start, stop = combine.obs_width, combine.obs_width + width
    units = getattr(rn, "scaled_units", "r")
    common = {"residual": residual, "warmup_scale": rn.warmup_scale}
    if residual == "additive":
        common.update(scale=combine.scale, table=combine.table, names=combine.phase_names)
    elif units == "sent":
        common.update(scaled_units="sent", base_cols=(start, stop))
    unit_name = "sent" if residual == "additive" or units == "sent" else "r"
    if rn.mode == "linear":
        a1 = _per_dim(rn.linear.a1, width, "residual_noise.linear.a1", device)
        a0 = _per_dim(rn.linear.a0, width, "residual_noise.linear.a0", device)
        print(f"[residual_noise] linear: s = |{a1.tolist()} * |base| + {a0.tolist()}| ({unit_name} units), "
              f"warm-up x{rn.warmup_scale}", flush=True)
        return LinearScale(start, stop, a1, a0, **common)
    fitted = load_std_mlp(rn.mlp.path)
    where = f"residual_noise.mlp.path {rn.mlp.path}"
    if fitted["input"] != rn.mlp.input:
        raise ValueError(f"{where} was fitted on the {fitted['input']!r} input; residual_noise.mlp.input is {rn.mlp.input!r}")
    if fitted["residual"] != residual:
        raise ValueError(f"{where} was fitted for the {fitted['residual']} residual; this run's is {residual}")
    if fitted["model"].out_dim != width:
        raise ValueError(f"{where} predicts {fitted['model'].out_dim} dims; the residual's action is {width} wide")
    check_raw_fingerprint(fitted["fingerprint"], base_fingerprint(cfg, combine), where,
                          "refit it with scripts/fit_noise_std.py")
    model = fitted["model"].to(device)
    print(f"[residual_noise] mlp ({rn.mlp.input} input, {rn.mlp.path}), gain {rn.mlp.gain}, warm-up x{rn.warmup_scale}",
          flush=True)
    return MlpScale(model, rn.mlp.input, rn.mlp.gain, combine.base_policy if rn.mlp.input == "base" else None, **common)
