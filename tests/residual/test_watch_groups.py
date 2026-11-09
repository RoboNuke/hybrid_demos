"""The watcher over a run's agents (my_project/residual_eval.py): which directories --run names, when a
checkpoint step is pending across agents, and how one FurnitureBench round's per-env data splits into each
agent's result. Pure functions: no sim, no wandb."""

from __future__ import annotations

import math
import os
import time

import pytest
import torch

import my_project.residual_eval as watch


# ---------------------------------------------------------------- agent_dirs
def test_an_agent_directory_is_one_slot(tmp_path):
    group = tmp_path / "g"
    agent = group / "g_a1"
    agent.mkdir(parents=True)
    assert watch.agent_dirs(agent, ["g_a0", "g_a1", "g_a2"]) == [agent]


def test_a_group_directory_is_every_agent_in_slot_order(tmp_path):
    group = tmp_path / "g"
    for name in ("g_a2", "g_a0", "g_a1"):
        (group / name).mkdir(parents=True)
    assert watch.agent_dirs(group, ["g_a0", "g_a1", "g_a2"]) == [group / "g_a0", group / "g_a1", group / "g_a2"]


def test_a_group_directory_missing_an_agent_fails_naming_it(tmp_path):
    group = tmp_path / "g"
    (group / "g_a0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as err:
        watch.agent_dirs(group, ["g_a0", "g_a1"])
    assert "g_a1" in str(err.value)


def test_a_run_without_agents_is_refused(tmp_path):
    with pytest.raises(ValueError):
        watch.agent_dirs(tmp_path, [])


# ---------------------------------------------------------------- checkpoint_dirs / pending_steps
def _write(path, age_seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))


def test_checkpoint_dirs_follow_the_routed_layout(tmp_path):
    agents = [tmp_path / "g_a0", tmp_path / "g_a1"]
    assert watch.checkpoint_dirs(agents, []) == [a / "checkpoints" for a in agents]
    assert watch.checkpoint_dirs(agents, ["insert", "screw"]) == [a / "insert" / "checkpoints" for a in agents]


def test_a_step_is_pending_only_once_every_agent_has_it_settled(tmp_path):
    dirs = [tmp_path / "g_a0" / "checkpoints", tmp_path / "g_a1" / "checkpoints"]
    settled = watch.SETTLED_SECONDS + 5
    for d in dirs:
        _write(d / "ckpt_10000.ckpt", settled)
        _write(d / "ckpt_best.ckpt", settled)  # never a numbered step
    _write(dirs[0] / "ckpt_20000.ckpt", settled)  # agent 1 has not written 20000 yet
    _write(dirs[0] / "ckpt_30000.ckpt", settled)
    _write(dirs[1] / "ckpt_30000.ckpt", 1.0)  # agent 1's file is still being written
    assert watch.pending_steps(dirs, [set(), set()]) == [10000]
    # recorded for every agent: no longer pending; recorded for one agent only: still pending
    assert watch.pending_steps(dirs, [{10000}, {10000}]) == []
    assert watch.pending_steps(dirs, [{10000}, set()]) == [10000]


def test_pending_steps_are_oldest_first(tmp_path):
    d = tmp_path / "g_a0" / "checkpoints"
    for step in (30000, 10000, 20000):
        _write(d / f"ckpt_{step}.ckpt", watch.SETTLED_SECONDS + 1)
    assert watch.pending_steps([d], [set()]) == [10000, 20000, 30000]


# ---------------------------------------------------------------- split_furniture
def _round(n, success, out, length, passed=None, correction=None):
    def pairs(items):
        idx = torch.tensor([i for i, _ in items], dtype=torch.long)
        val = torch.tensor([v for _, v in items], dtype=torch.float32)
        return idx, val

    return {
        "success": torch.tensor(success, dtype=torch.bool),
        "out": torch.tensor(out, dtype=torch.bool),
        "length": torch.tensor(length, dtype=torch.long),
        "passed": {k: pairs(v) for k, v in (passed or {}).items()},
        "correction": {k: pairs(v) for k, v in (correction or {}).items()},
    }


def test_each_agent_gets_its_own_env_block():
    # 4 envs, 2 agents: agent 0 owns envs 0-1, agent 1 owns envs 2-3
    data = _round(4, success=[1, 0, 0, 0], out=[0, 0, 1, 1], length=[10, 20, 30, 40])
    a0 = watch.split_furniture([data], range(0, 2), None)
    a1 = watch.split_furniture([data], range(2, 4), None)
    assert (a0["success_rate"], a0["out_of_bounds_rate"], a0["length_mean"], a0["episodes"]) == (0.5, 0.0, 15.0, 2)
    assert (a1["success_rate"], a1["out_of_bounds_rate"], a1["lengths"], a1["episodes"]) == (0.0, 1.0, [30, 40], 2)


def test_rounds_concatenate_and_trim_to_num_rollouts():
    r1 = _round(2, success=[1, 1], out=[0, 0], length=[1, 2])
    r2 = _round(2, success=[0, 0], out=[0, 0], length=[3, 4])
    r3 = _round(2, success=[1, 1], out=[0, 0], length=[5, 6])
    # 5 episodes over rounds of 2: all of r1 and r2, the first env of r3
    result = watch.split_furniture([r1, r2, r3], range(0, 2), 5)
    assert result["lengths"] == [1, 2, 3, 4, 5] and result["episodes"] == 5
    assert result["success_rate"] == pytest.approx(3 / 5)


def test_phase_and_correction_values_follow_their_env():
    nan = float("nan")
    data = _round(
        4,
        success=[0, 0, 0, 0], out=[0, 0, 0, 0], length=[1, 1, 1, 1],
        # env 0 reached grasp and passed, env 1 never reached it, env 2 reached and failed, env 3 passed
        passed={"phase/passed_grasp": [(0, 1.0), (1, nan), (2, 0.0), (3, 1.0)],
                "phase/passed_success": [(0, nan), (1, nan), (2, 1.0), (3, 0.0)]},
        correction={"residual_mm/insert": [(0, 2.0), (0, 4.0), (3, 10.0)]},
    )
    a0 = watch.split_furniture([data], range(0, 2), None)
    a1 = watch.split_furniture([data], range(2, 4), None)
    assert a0["phase_success/grasp"] == 1.0 and a0["phase_reached/grasp"] == 1
    assert "phase_success/screw" not in a0  # no episode of agent 0 reached it
    assert a0["residual_mm/insert"] == pytest.approx(3.0)
    assert a1["phase_success/grasp"] == 0.5 and a1["phase_reached/grasp"] == 2
    assert a1["phase_success/screw"] == 0.5 and a1["phase_reached/screw"] == 2
    assert a1["residual_mm/insert"] == pytest.approx(10.0)


def test_trimming_drops_the_values_of_untaken_envs():
    data = _round(
        2, success=[1, 1], out=[0, 0], length=[7, 8],
        passed={"phase/passed_grasp": [(0, 1.0), (1, 0.0)]},
        correction={"residual_deg/corner": [(1, 99.0)]},
    )
    # one episode wanted: env 0 only, so env 1's phase and correction values are left out
    result = watch.split_furniture([data], range(0, 2), 1)
    assert result["lengths"] == [7] and result["phase_success/grasp"] == 1.0
    assert "residual_deg/corner" not in result


def test_the_round_count_covers_num_rollouts():
    assert math.ceil(200 / 40) == 5  # the ablation config: 5 agents x 40 envs, 200 episodes each
    assert math.ceil(100 / 100) == 1  # residual_watch_furniture: one round, as before


# ---------------------------------------------------------------- new_run_id
def test_a_new_run_id_has_wandbs_form_with_and_without_the_helper(monkeypatch):
    import re

    import wandb

    assert not hasattr(wandb.util, "generate_id")  # gone since wandb 0.30: the watcher must not call it
    first = watch.new_run_id()
    assert re.fullmatch(r"[a-z0-9]{8}", first)
    import wandb.sdk.lib.runid as runid

    monkeypatch.delattr(runid, "generate_id")  # the fallback has the same form
    second = watch.new_run_id()
    assert re.fullmatch(r"[a-z0-9]{8}", second) and second != first
