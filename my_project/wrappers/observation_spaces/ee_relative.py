"""``ee_relative.*``: the observation relative to the EE, in the EE's own axes (:class:`EERelative`, the
:class:`~my_project.wrappers.chain.ConversionWrapper` around it). Every listed pose becomes ``(R_ee^T (p - p_ee), R_ee^T
R)``, written back in its own columns and parametrisation; every listed vector ``R_ee^T v``; the EE's own columns are
removed (in its own frame it is the origin). Applied to the agent's and the critics' observation columns alike;
columns past the listed ones (a tracked target appended below, say) are left as they are. The action is untouched."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch

from my_project.wrappers.chain import Conversion, ConversionWrapper
from my_project.wrappers.pose_math import mat_to_rot, mat_to_rot6d, relative_pose, rot_to_mat

__all__ = ["EERelative", "EERelativeWrapper"]


class EERelative(Conversion):
    """``ee``: ``(pos span, rot span, rot param)`` of the EE; ``poses``: the same for every pose re-expressed;
    ``vectors``: 3-wide spans rotated into the EE's axes."""

    def __init__(self, ee: Tuple[Sequence[int], Sequence[int], str], poses=(), vectors=()) -> None:
        self.ee = (slice(*map(int, ee[0])), slice(*map(int, ee[1])), str(ee[2]))
        self.poses = [(slice(*map(int, p)), slice(*map(int, r)), str(param)) for p, r, param in poses]
        self.vectors = [slice(*map(int, v)) for v in vectors]

    def build(self, obs_width: int, critic_width: int, action_dim: int) -> None:
        spans = [self.ee[0], self.ee[1]] + [s for p, r, _ in self.poses for s in (p, r)] + self.vectors
        last = max(s.stop for s in spans)
        if last > min(obs_width, critic_width):
            raise ValueError(f"ee_relative's columns run to {last}, past the observation ({obs_width}) or the critics' "
                             f"({critic_width})")
        gone = set(range(self.ee[0].start, self.ee[0].stop)) | set(range(self.ee[1].start, self.ee[1].stop))
        self.gone = gone
        self.obs_width, self.critic_width, self.action_dim = obs_width - len(gone), critic_width - len(gone), action_dim
        self._keep = {}

    def _express(self, x: torch.Tensor) -> torch.Tensor:
        pos, rot, param = self.ee
        p_ee, R_ee = x[..., pos], rot_to_mat(x[..., rot], param)
        out = x.clone()
        for p, r, rparam in self.poses:
            rp, rR = relative_pose(x[..., p], rot_to_mat(x[..., r], rparam), p_ee, R_ee)
            out[..., p] = rp
            out[..., r] = mat_to_rot6d(rR) if rparam == "rot6d" else mat_to_rot(rR, rparam)
        Rt = R_ee.transpose(-1, -2)
        for v in self.vectors:
            out[..., v] = (Rt @ x[..., v].unsqueeze(-1)).squeeze(-1)
        width = x.shape[-1]
        keep = self._keep.get((width, x.device))
        if keep is None:
            keep = torch.tensor([c for c in range(width) if c not in self.gone], dtype=torch.long, device=x.device)
            self._keep[(width, x.device)] = keep
        return out.index_select(-1, keep)

    def observe(self, policy, critic, base, state, *, fresh, new_chunk):
        return self._express(policy), self._express(critic), base, {}, state

    def act(self, action, ctx, state):
        return action, state

    def invert(self, executed, ctx, state):
        return executed, state


class EERelativeWrapper(ConversionWrapper):
    def __init__(self, env: Any, cfg: Any) -> None:
        er = cfg.ee_relative
        conversion = EERelative((list(er.ee_pos), list(er.ee_rot), er.ee_rot_param),
                                [(list(p.pos), list(p.rot), p.param) for p in er.poses], [list(v) for v in er.vectors])
        super().__init__(env, conversion)
        print(f"[ee_relative] observation relative to the EE (columns {list(er.ee_pos)} + {list(er.ee_rot)} "
              f"{er.ee_rot_param}, removed): {len(er.poses)} poses and {len(er.vectors)} vectors re-expressed; "
              f"observation {self.obs_width}", flush=True)
