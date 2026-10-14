"""The diffusion-policy noise-prediction 1-D temporal U-Net.

A port of robust-rearrangement's ``src/models/unet.py``
(https://github.com/ankile/robust-rearrangement/blob/main/src/models/unet.py, MIT), itself
Diffusion Policy's ``ConditionalUnet1D`` (Chi et al., RSS 2023,
https://github.com/real-stanford/diffusion_policy, MIT): residual Conv1d blocks over the
action sequence, FiLM-conditioned on ``[timestep embedding, global_cond]``, three levels by
default (``down_dims`` [256, 512, 1024] -- ResiP's state-based one_leg model, 66M parameters
at its widths). It is what ResiP's reported state BC numbers come from (paper Table IV).

Departures from the reference: the debugger import is dropped; ``SinusoidalPosEmb`` comes
from the transformer module (the same function); ``global_cond`` may be ``(B, T_o, D)`` and
is flattened here (the reference flattens it in the policy, ``flatten_obs: True``).
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn

from .transformer import SinusoidalPosEmb

__all__ = ["ConditionalUnet1D"]


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks; FiLM (per-channel scale and bias from the conditioning) between them."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, n_groups: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2), nn.Unflatten(-1, (-1, 1)))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` ``(B, in_channels, T)``, ``cond`` ``(B, cond_dim)`` -> ``(B, out_channels, T)``."""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(x.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0, ...] * out + embed[:, 1, ...]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """``forward(sample (B, T, input_dim), timestep, global_cond) -> (B, T, input_dim)``.

    ``global_cond_dim`` is ``n_obs_steps * obs_dim``; ``down_dims`` sets the levels (each but
    the last halves the sequence, so ``horizon`` must be divisible by ``2 ** (len - 1)``).
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed)
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        def block(a: int, b: int) -> ConditionalResidualBlock1D:
            return ConditionalResidualBlock1D(a, b, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups)

        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([block(dim_in, dim_out), block(dim_out, dim_out), Downsample1d(dim_out) if not is_last else nn.Identity()])
            )
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)  # as the reference: every up level upsamples
            self.up_modules.append(
                nn.ModuleList([block(dim_out * 2, dim_in), block(dim_in, dim_in), Upsample1d(dim_in) if not is_last else nn.Identity()])
            )
        self.final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size), nn.Conv1d(start_dim, input_dim, 1))

    def forward(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], global_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = sample.moveaxis(-1, -2)  # (B, C, T)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            feature = torch.cat([feature, global_cond.reshape(sample.shape[0], -1)], dim=-1)

        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, feature), feature)
            skips.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = upsample(resnet2(resnet(x, feature), feature))
        return self.final_conv(x).moveaxis(-1, -2)
