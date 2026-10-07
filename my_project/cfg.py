"""The project's config sections. Registered in the entry scripts' ``setup`` with
``register_section("demo", DemoCfg)``; experiment YAMLs then set ``demo.*`` fields and
the resolved config records them like any package section. Follow the package CLAUDE.md's
"Add a config class" recipe (OmegaConf-supported types, validate() for cross-field rules,
every field documented in this repo's README)."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DemoCfg:
    """Example section -- replace with what your project actually configures."""

    #: example field, read by your project code via ``cfg.demo.reward_scale``
    reward_scale: float = 1.0

    def validate(self, cfg: Any) -> None:
        if self.reward_scale <= 0:
            raise ValueError(f"demo.reward_scale must be > 0, got {self.reward_scale}")
