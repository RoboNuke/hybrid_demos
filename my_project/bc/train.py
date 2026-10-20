"""Behaviour-clone the diffusion policy from demonstrations.

    python scripts/train_bc.py --config configs/experiments/diffusion_bc.yaml --headless

A port of robust-rearrangement's ``src/train/bc.py`` (state observations, one GPU) onto this
project's config and logging. One epoch = ``bc.steps_per_epoch`` gradient steps; the run is
``trainer.total_timesteps`` gradient steps. Published once per epoch, at ``step=`` the
gradient-step count, to one wandb run (``experiment.num_agents`` must be 1):

* every epoch: ``epoch``, ``epoch_loss``, ``train_bc_loss``, ``train_grad_norm`` (the epoch
  mean; the reference logged only the last batch's), ``actor_lr``;
* every ``bc.eval_every`` epochs: ``test_epoch_loss``, ``test_bc_loss`` and
  ``early_stopper/{counter,best_loss,ema_loss}``;
* on eval epochs that are also ``bc.sample_every`` epochs: DDIM-sampled chunks against the
  demo actions, ``action_sample/{train,val}_action_mse_error`` plus one
  ``..._<group>`` per ``bc.action_metric_groups`` entry;
* every ``bc.rollout_every`` epochs (eval epochs, once the test loss is under
  ``bc.rollout_loss_threshold``): ``success_rate``, ``best_success_rate``,
  ``epoch_mean_return``, ``n_success``, ``n_rollouts``, plus the package eval summary under
  ``rollout/``.

Training stops early after ``bc.max_hours`` of wall-clock time, if set.

Checkpoints go to ``<run dir>/checkpoints/`` and each one is uploaded to the run's files as
it is written (same name, so a rewrite replaces it): ``actor_chkpt_last.pt`` every epoch,
``actor_chkpt_best_test_loss.pt``, ``actor_chkpt_best_success_rate.pt``, and
``actor_chkpt_<epoch>.pt`` every ``trainer.checkpoint_interval`` gradient steps. With
``bc.save_only_best_ckpt`` only ``actor_chkpt_best_success_rate.pt`` is written and uploaded
(at the first rollout, then on every improvement). The run summary holds the dataset sizes.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from robonuke_rl_core.config import add_config_args, dump, load_from_args


class EarlyStopper:
    """The reference's stopper: patience counted on an EMA of the test loss."""

    def __init__(self, patience: float, smooth_factor: float) -> None:
        self.patience = float(patience)
        self.smooth_factor = smooth_factor
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.ema_loss: Optional[float] = None

    def update(self, val_loss: float) -> bool:
        if self.ema_loss is None:
            self.ema_loss = val_loss
        else:
            self.ema_loss = self.smooth_factor * self.ema_loss + (1 - self.smooth_factor) * val_loss
        if self.best_loss is None:
            self.best_loss = self.ema_loss
        if self.ema_loss < self.best_loss:
            self.best_loss = self.ema_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def action_mse_metrics(split: str, predicted, target, groups: Dict[str, list]) -> Dict[str, Any]:
    """``action_sample/<split>_action_mse_error[_<group>]`` in action units, as tensors."""
    import torch.nn.functional as F

    prefix = f"action_sample/{split}_action_mse_error"
    out = {prefix: F.mse_loss(predicted, target)}
    for name, dims in groups.items():
        out[f"{prefix}_{name}"] = F.mse_loss(predicted[..., dims], target[..., dims])
    return out


def rollout_metrics(cfg, env, rollout_policy, best_success_rate: float) -> Dict[str, Any]:
    """Roll the policy out with the package's eval accounting; the reference's keys.

    ``eval.num_rollouts`` episodes, first episode per env per round, ``experiment.seed``.
    Success is the package task-metrics wrapper's ``episode/success`` (on by default for
    Forge tasks); without it there is no success rate to report, so this raises.
    """
    from robonuke_rl_core.evaluation import run_eval

    result = run_eval(
        env=env,
        policy=rollout_policy,
        num_rollouts=cfg.eval.num_rollouts,
        max_episode_length=int(env.unwrapped.max_episode_length),
        seed=cfg.experiment.seed,
        save_state=False,
        device=env.device,
    )
    summary = result.summary
    if "episode/success/mean" not in summary:
        raise KeyError(
            "rollouts produced no 'episode/success' metric: keep wrappers.task_metrics.enabled "
            f"on (it needs a Forge-family task); summary keys: {sorted(summary)}"
        )
    rate, episodes = summary["episode/success/mean"], summary["episodes"]
    out = {
        "success_rate": rate,
        "best_success_rate": max(best_success_rate, rate),
        "epoch_mean_return": summary["return/mean"],
        "n_success": round(rate * episodes),
        "n_rollouts": episodes,
    }
    out.update({f"rollout/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
    return {k: float(v) for k, v in out.items()}


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    app_launcher = AppLauncher(args)
    try:
        return _train(args, overrides, setup, app_launcher)
    except BaseException:
        # Kit's shutdown turns any uncaught exception into exit status 0, so a crashed run
        # would look finished to a job chain. Report it and leave with a failing status.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _train(args, overrides, setup, app_launcher) -> int:
    if setup is not None:
        setup()

    import numpy as np
    import torch
    from diffusers.optimization import get_scheduler
    from omegaconf import OmegaConf

    from robonuke_rl_core.learners.base import run_dirs
    from robonuke_rl_core.logging import WandbLogger

    from my_project.bc.dataset import WindowedSplit, load_lerobot, split_episodes
    from my_project.cfg import BC_LEARNER
    from my_project.models.diffusion_policy import DiffusionPolicy

    cfg = load_from_args(args, overrides)
    if cfg.trainer.learner != BC_LEARNER:
        raise ValueError(
            f"scripts/train_bc.py runs trainer.learner: {BC_LEARNER}, got "
            f"{cfg.trainer.learner!r} (use scripts/train.py for sac / ppo)"
        )
    bc, actor_cfg = cfg.bc, cfg.model.actor
    seed = cfg.experiment.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device if getattr(args, "device", None) else "cuda:0")

    # ------------------------------------------------------------------ data
    demos = load_lerobot(bc.dataset, bc.obs_key, bc.action_key)
    for group, dims in bc.action_metric_groups.items():
        if max(dims) >= demos.action_dim:
            raise ValueError(
                f"bc.action_metric_groups.{group} names dims {list(dims)}, but the dataset's "
                f"actions have {demos.action_dim} dims"
            )
    train_eps, test_eps = split_episodes(demos.num_episodes, bc.num_demos, bc.test_split, seed)
    used = demos.frames_of(np.concatenate([train_eps, test_eps]))
    policy = DiffusionPolicy(actor_cfg, demos.obs_dim, demos.action_dim).to(device)
    policy.normalizer.fit("obs", demos.obs[used].to(device))
    policy.normalizer.fit("action", demos.action[used].to(device))
    if bc.normalization == "per_unit":  # one range per physical unit, obs and action together
        from my_project.families import action_units, obs_units

        policy.normalizer.fit_units(
            demos.obs[used].to(device), demos.action[used].to(device),
            obs_units(cfg.task_name, cfg.task_cfg), action_units(cfg.task_name),
        )
    window = dict(
        n_obs_steps=actor_cfg.n_obs_steps,
        horizon=actor_cfg.horizon,
        action_horizon=actor_cfg.action_horizon,
        device=device,
    )
    train_data = WindowedSplit(demos, train_eps, policy.normalizer, **window)
    test_data = WindowedSplit(demos, test_eps, policy.normalizer, **window)
    print(
        f"[train_bc] {demos.source}: {demos.num_episodes} episodes, obs {demos.obs_dim}, "
        f"action {demos.action_dim}; windows train {len(train_data)} / test {len(test_data)}",
        flush=True,
    )

    # ------------------------------------------------------------------ env (rollouts only)
    group_dir = Path(cfg.trainer.output_dir) / cfg.wandb.project / cfg.wandb.group
    env = None
    if bc.rollout_every > 0:
        import gymnasium as gym

        from robonuke_rl_core.envs.build import build_env, prepare_task, wrap_for_skrl

        prepare_task(cfg, cfg.task_name, cfg.task_cfg)
        raw = gym.make(cfg.task_name, cfg=cfg.task_cfg)
        config_path = dump(cfg, group_dir, raw.unwrapped.cfg)
        # build_env applies the project's wrappers too (register_env_wrapper in setup)
        env = wrap_for_skrl(build_env(cfg, raw, cfg.task_name), cfg.experiment.seed)
        env_obs = int(np.prod(env.observation_space.shape))
        env_act = int(np.prod(env.action_space.shape))
        if (env_obs, env_act) != (demos.obs_dim, demos.action_dim):
            raise ValueError(
                f"rollouts need the env and the demos to agree: env obs {env_obs} / action "
                f"{env_act}, demos {bc.obs_key} {demos.obs_dim} / {bc.action_key} "
                f"{demos.action_dim}. Make the task's observations match the dataset layout, "
                "or set bc.rollout_every=0 to train without rollouts."
            )
    else:
        config_path = dump(cfg, group_dir, cfg.task_cfg)  # no env: the cfg it would get
    print(f"[train_bc] config: {config_path}", flush=True)

    # ------------------------------------------------------------------ optimizer
    total_steps = cfg.trainer.total_timesteps
    num_epochs = total_steps // bc.steps_per_epoch
    params = list(policy.model.parameters())
    optimizer = torch.optim.AdamW(params, lr=bc.lr, weight_decay=bc.weight_decay)
    lr_scheduler = get_scheduler(
        name=bc.lr_schedule,
        optimizer=optimizer,
        num_warmup_steps=bc.warmup_steps,
        num_training_steps=total_steps,
    )
    max_norm = bc.max_grad_norm if bc.max_grad_norm > 0 else math.inf
    test_batches = max(int(round(bc.steps_per_epoch * bc.test_split)), 1)

    # ------------------------------------------------------------------ logging
    logger = WandbLogger.from_config(
        cfg,
        OmegaConf.to_container(OmegaConf.load(config_path), resolve=True),
        config_path=config_path,
        device=device,
    )
    run = logger.runs[0]
    run.summary.update(
        {
            "num_samples_train": len(train_data),
            "num_samples_test": len(test_data),
            "num_episodes_train": train_data.num_episodes,
            "num_episodes_test": test_data.num_episodes,
            "num_episodes_dataset": demos.num_episodes,
            "train_episode_indices": train_eps.tolist(),
            "test_episode_indices": test_eps.tolist(),
            "dataset": bc.dataset,
            "dataset_source": demos.source,
            "obs_dim": demos.obs_dim,
            "action_dim": demos.action_dim,
            "num_parameters": sum(p.numel() for p in params),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    ckpt_dir = run_dirs(cfg)[0] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str, state: Dict[str, Any]) -> None:
        path = ckpt_dir / name
        torch.save(state, path)
        logger.save_file(path, agent=0)

    class RolloutPolicy:
        """``run_eval``'s policy: chunked acting, reset at every round."""

        def __call__(self, observations, states):
            return policy.act(observations)

        def reset(self):
            policy.reset()

    def checkpoint_state() -> Dict[str, Any]:
        """The policy (network + normalizer stats) and everything needed to resume."""
        return {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_test_loss": best_test_loss,
            "best_success_rate": best_success_rate,
            "obs_dim": demos.obs_dim,
            "action_dim": demos.action_dim,
        }

    def scalar(value) -> torch.Tensor:
        """A host number as a device tensor: the accumulator sums everything on ``device``."""
        return torch.tensor(float(value), device=device)

    stopper = EarlyStopper(bc.early_stop_patience, bc.early_stop_smooth)
    best_test_loss, best_success_rate = math.inf, 0.0
    saved_best = False
    global_step = 0
    print(
        f"[train_bc] {num_epochs} epochs x {bc.steps_per_epoch} steps, batch {bc.batch_size}; "
        f"wandb: mode={cfg.wandb.mode} group={cfg.wandb.group}",
        flush=True,
    )

    training_started = time.time()
    try:
        for epoch in range(num_epochs):
            started = time.time()
            policy.train()
            for _ in range(bc.steps_per_epoch):
                batch = train_data.batch(bc.batch_size)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = policy.compute_loss(batch["obs"], batch["action"])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
                optimizer.step()
                lr_scheduler.step()
                global_step += 1
                logger(
                    0,
                    {
                        "epoch_loss": loss.detach(),
                        **{f"train_{k}": v for k, v in parts.items()},
                        "train_grad_norm": grad_norm.detach(),
                    },
                    global_step,
                )
            epoch_log: Dict[str, Any] = {
                "epoch": scalar(epoch),
                "actor_lr": scalar(optimizer.param_groups[0]["lr"]),
            }

            stop = False
            is_eval = (epoch + 1) % bc.eval_every == 0
            if is_eval:
                policy.eval()
                test_sum, test_parts = 0.0, {}
                with torch.no_grad():
                    for _ in range(test_batches):
                        batch = test_data.batch(bc.batch_size)
                        loss, parts = policy.compute_loss(batch["obs"], batch["action"])
                        test_sum += loss
                        for k, v in parts.items():
                            test_parts[k] = test_parts.get(k, 0.0) + v
                test_loss = float(test_sum / test_batches)
                epoch_log["test_epoch_loss"] = scalar(test_loss)
                for k, v in test_parts.items():
                    epoch_log[f"test_{k}"] = v / test_batches

                if (
                    env is not None
                    and (epoch + 1) % bc.rollout_every == 0
                    and test_loss < bc.rollout_loss_threshold
                ):
                    epoch_log.update(
                        {k: scalar(v) for k, v in rollout_metrics(cfg, env, RolloutPolicy(), best_success_rate).items()}
                    )
                    policy.reset()
                    rate = float(epoch_log["success_rate"])
                    # only-best keeps a checkpoint from the first rollout on, even at 0%
                    first = bc.save_only_best_ckpt and not saved_best
                    if rate > best_success_rate or first:
                        best_success_rate = max(rate, best_success_rate)
                        save("actor_chkpt_best_success_rate.pt", checkpoint_state())
                        saved_best = True

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    if not bc.save_only_best_ckpt:
                        save("actor_chkpt_best_test_loss.pt", checkpoint_state())

                if (epoch + 1) % bc.sample_every == 0:
                    for split, data in (("train", train_data), ("val", test_data)):
                        batch = data.batch(bc.batch_size)
                        predicted = policy.predict_chunk(batch["obs"])
                        target = policy.normalizer.denormalize(batch["action"], "action")
                        epoch_log.update(
                            action_mse_metrics(split, predicted, target, bc.action_metric_groups)
                        )

                stop = stopper.update(test_loss)
                epoch_log["early_stopper/counter"] = scalar(stopper.counter)
                epoch_log["early_stopper/best_loss"] = scalar(stopper.best_loss)
                epoch_log["early_stopper/ema_loss"] = scalar(stopper.ema_loss)

            if not bc.save_only_best_ckpt:
                interval = cfg.trainer.checkpoint_interval
                if interval and global_step % interval == 0:
                    save(f"actor_chkpt_{epoch}.pt", checkpoint_state())
                save("actor_chkpt_last.pt", checkpoint_state())

            logger(0, epoch_log, global_step)
            logger.flush(global_step)
            print(
                f"[train_bc] epoch {epoch + 1}/{num_epochs} step {global_step} "
                f"({time.time() - started:.1f}s)"
                + (f" test {test_loss:.4f}" if is_eval else ""),
                flush=True,
            )
            if stop:
                print(f"[train_bc] early stop at epoch {epoch + 1}", flush=True)
                break
            hours = (time.time() - training_started) / 3600.0
            if bc.max_hours is not None and hours >= bc.max_hours:
                print(f"[train_bc] time limit: {hours:.2f} h >= bc.max_hours {bc.max_hours} at epoch {epoch + 1}", flush=True)
                break
    finally:
        logger.close()

    if env is not None:
        env.close()
    app_launcher.app.close()
    return 0
