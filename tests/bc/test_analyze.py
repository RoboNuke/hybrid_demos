"""The offline chunk analysis on a furniture FBAbs run: its own swap groups and the
leg-near-the-top error (no sim; a tiny untrained U-Net on a synthetic 61/10 dataset)."""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from my_project.bc.analyze import analyze
from my_project.models.cfg import DiffusionUnetActorCfg
from my_project.models.diffusion_policy import DiffusionPolicy

OBS, ACT = 61, 10


def write_dataset(root, episodes=4, length=12):
    rng = np.random.default_rng(0)
    rows = {"episode_index": [], "frame_index": [], "observation.state_abs": [], "action.abs": []}
    for e in range(episodes):
        for f in range(length):
            obs = rng.normal(size=OBS).astype(np.float32) * 0.01
            obs[16:19] = [0.0, 0.24, 0.0]  # the top
            # the leg walks from far (26 cm) to the top over the episode: some windows near it
            obs[52:55] = [0.26 * (1 - f / (length - 1)), 0.24, 0.0]
            rows["episode_index"].append(e)
            rows["frame_index"].append(f)
            rows["observation.state_abs"].append(obs.tolist())
            rows["action.abs"].append(rng.normal(size=ACT).astype(np.float32).tolist())
    table = pa.table(
        {
            "episode_index": pa.array(rows["episode_index"], pa.int64()),
            "frame_index": pa.array(rows["frame_index"], pa.int64()),
            "observation.state_abs": pa.array(rows["observation.state_abs"], pa.list_(pa.float32(), OBS)),
            "action.abs": pa.array(rows["action.abs"], pa.list_(pa.float32(), ACT)),
        }
    )
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return np.array(rows["observation.state_abs"]), np.array(rows["action.abs"])


def test_the_furniture_report_has_its_groups_and_the_near_top_error(tmp_path):
    torch.manual_seed(0)
    data = tmp_path / "data"
    obs, act = write_dataset(data)
    actor = DiffusionUnetActorCfg(
        horizon=8, n_obs_steps=2, action_horizon=4, down_dims=[16, 32], diffusion_step_embed_dim=16,
        n_groups=4, inference_steps=2, num_diffusion_iters=10, warm_start=False,
    )
    policy = DiffusionPolicy(actor, OBS, ACT)
    policy.normalizer.fit("obs", torch.tensor(obs))
    policy.normalizer.fit("action", torch.tensor(act))
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    torch.save({"obs_dim": OBS, "action_dim": ACT, "model_state_dict": policy.state_dict()},
               run / "checkpoints" / "best.pt")
    OmegaConf.save(
        OmegaConf.create({
            "task": {"name": "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0"},
            "experiment": {"seed": 0},
            "model": {"architecture": "diffusion_unet", "actor": dataclasses.asdict(actor)},
            "bc": {"dataset": str(data), "obs_key": "observation.state_abs", "action_key": "action.abs",
                   "num_demos": 3, "test_split": 0.25},
        }),
        run / "resolved_config.yaml",
    )
    args = argparse.Namespace(run=str(run), checkpoint="best.pt", dataset=None, windows=32, samples=2, plots=1,
                              inference_steps=0, seed=0, device="cpu", out=str(tmp_path / "out"))
    report = analyze(args)
    train = report["train"]
    assert {"swap_ee_pos_moves_mm", "swap_top_pose_moves_mm", "swap_leg_pose_moves_mm"} <= set(train)
    assert "swap_fingertip_estimate_moves_mm" not in train
    assert 0 < train["near_top_windows"] < train["windows"]
    assert train["near_top_error_mm_step0"] > 0
