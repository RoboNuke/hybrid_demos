"""Demonstrations: windows match the reference's sampler, the split follows bc.num_demos."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from my_project.bc.dataset import WindowedSplit, load_lerobot, split_episodes
from my_project.models.diffusion_policy import MinMaxNormalizer

OBS_DIM, ACT_DIM = 3, 2


# ------------------------------------------------------------------ the reference, verbatim
def create_sample_indices(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i])
    return np.array(indices)


def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = torch.zeros(size=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# ------------------------------------------------------------------ a tiny LeRobot dataset
def write_dataset(root, lengths, shuffle_rows=False):
    rows = {"episode_index": [], "frame_index": [], "observation.state": [], "action": []}
    value = 0.0
    for episode, length in enumerate(lengths):
        for frame in range(length):
            rows["episode_index"].append(episode)
            rows["frame_index"].append(frame)
            rows["observation.state"].append([value + d for d in range(OBS_DIM)])
            rows["action"].append([-value - d for d in range(ACT_DIM)])
            value += 1.0
    table = pa.table(
        {
            "episode_index": pa.array(rows["episode_index"], pa.int64()),
            "frame_index": pa.array(rows["frame_index"], pa.int64()),
            "observation.state": pa.array(rows["observation.state"], pa.list_(pa.float32(), OBS_DIM)),
            "action": pa.array(rows["action"], pa.list_(pa.float32(), ACT_DIM)),
        }
    )
    if shuffle_rows:
        table = table.take(np.random.default_rng(0).permutation(table.num_rows))
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return root


@pytest.mark.parametrize("n_obs_steps,horizon,action_horizon", [(1, 4, 2), (2, 4, 3), (3, 2, 1)])
def test_windows_match_the_reference_sampler(tmp_path, n_obs_steps, horizon, action_horizon):
    lengths = [7, 5, 9]
    demos = load_lerobot(str(write_dataset(tmp_path, lengths, shuffle_rows=True)), "observation.state", "action")
    normalizer = MinMaxNormalizer(OBS_DIM, ACT_DIM)  # identity-ish stats are fine here
    split = WindowedSplit(
        demos, np.arange(len(lengths)), normalizer,
        n_obs_steps=n_obs_steps, horizon=horizon, action_horizon=action_horizon, device="cpu",
    )

    sequence = n_obs_steps + horizon - 1
    indices = create_sample_indices(
        np.cumsum(lengths), sequence, pad_before=n_obs_steps - 1, pad_after=action_horizon - 1
    )
    assert len(split) == len(indices)
    data = {"obs": normalizer.normalize(demos.obs, "obs"), "action": normalizer.normalize(demos.action, "action")}
    ours = split.windows(torch.arange(len(split)))
    for row, (b0, b1, s0, s1, _) in enumerate(indices):
        ref = sample_sequence(data, sequence, b0, b1, s0, s1)
        assert torch.equal(ours["obs"][row], ref["obs"][:n_obs_steps]), row
        assert torch.equal(ours["action"][row], ref["action"][n_obs_steps - 1 : n_obs_steps - 1 + horizon]), row


def test_loading_rejects_gaps_and_missing_columns(tmp_path):
    root = write_dataset(tmp_path / "a", [3, 3])
    with pytest.raises(KeyError, match="nope"):
        load_lerobot(str(root), "nope", "action")
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    pq.write_table(table.filter(pa.compute.not_equal(table["index"] if "index" in table.column_names else table["frame_index"], 1)), path)
    with pytest.raises(ValueError, match="frame_index"):
        load_lerobot(str(root), "observation.state", "action")


def test_batches_walk_a_shuffled_pass_before_repeating(tmp_path):
    demos = load_lerobot(str(write_dataset(tmp_path, [6, 6])), "observation.state", "action")
    split = WindowedSplit(demos, np.arange(2), MinMaxNormalizer(OBS_DIM, ACT_DIM),
                          n_obs_steps=1, horizon=2, action_horizon=1, device="cpu")
    torch.manual_seed(0)
    assert len(split) == 10
    seen = torch.cat([split.batch(5)["obs"][:, 0, 0] for _ in range(2)])
    assert sorted(seen.tolist()) == sorted(split.windows(torch.arange(len(split)))["obs"][:, 0, 0].tolist())


# ------------------------------------------------------------------ the split
def test_num_demos_trains_on_that_many_and_tests_on_unused_episodes():
    train, test = split_episodes(2221, 50, 0.05, seed=3)
    assert len(train) == 50 and len(test) == math.ceil(0.05 * 50) == 3
    assert not set(train) & set(test)
    again = split_episodes(2221, 50, 0.05, seed=3)
    assert np.array_equal(train, again[0]) and np.array_equal(test, again[1])
    assert not np.array_equal(train, split_episodes(2221, 50, 0.05, seed=4)[0])


@pytest.mark.parametrize("num_demos", [None, 100])
def test_using_every_episode_holds_the_test_set_out_of_them(num_demos):
    train, test = split_episodes(100, num_demos, 0.05, seed=0)
    assert len(test) == 5 and len(train) == 95
    assert sorted(set(train) | set(test)) == list(range(100))


def test_impossible_splits_raise():
    with pytest.raises(ValueError, match="num_demos=101"):
        split_episodes(100, 101, 0.05, seed=0)
    with pytest.raises(ValueError, match="unused"):
        split_episodes(100, 99, 0.05, seed=0)  # 99 + ceil(4.95)=5 > 100
