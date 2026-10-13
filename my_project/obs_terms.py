"""Extra Factory / Forge observation terms, usable by name in ``task.cfg.obs_order``.

Factory and Forge build their observation as a dict of named terms and flatten it with
``factory_utils.collapse_obs_dict(obs_dict, order)``, sizing it from the ``OBS_DIM_CFG`` table.
:func:`install_obs_terms` adds terms to both, the same way the package adds its rot6d terms, so
the observation layout stays pure config (``task.cfg.obs_order``):

* ``hole_estimate`` (3): the env's noisy estimate of the hole top, env (robot base) frame:
  ``fingertip_pos - fingertip_pos_rel_fixed`` from the same dict. In the policy dict both are
  noisy (Forge: ``noisy_fingertip_pos`` and ``noisy_fingertip_pos - noisy_fixed_pos``), so this
  is exactly ``fixed_pos_obs_frame + init_fixed_pos_obs_noise``, the estimate Forge's relative
  position and actions use. (In the critic dict both are true, so there it is the true hole top.)

Forge's own ``fingertip_pos`` term is already the noisy fingertip estimate in the policy dict.
Call once, in the entry scripts' ``setup()``, before the env is made. Idempotent.
"""

from __future__ import annotations

__all__ = ["install_obs_terms", "EXTRA_TERMS"]

#: term name -> width
EXTRA_TERMS = {"hole_estimate": 3}


def install_obs_terms() -> None:
    from isaaclab_tasks.direct.factory import factory_utils
    from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG

    for table in (OBS_DIM_CFG, STATE_DIM_CFG):
        for name, width in EXTRA_TERMS.items():
            if table.get(name, width) != width:
                raise ValueError(f"observation term {name!r} is already registered with width {table[name]}")
            table[name] = width
    if getattr(factory_utils, "_project_terms_installed", False):
        return
    original = factory_utils.collapse_obs_dict

    def collapse(obs_dict, obs_order):
        if "fingertip_pos" in obs_dict and "fingertip_pos_rel_fixed" in obs_dict:
            obs_dict.setdefault("hole_estimate", obs_dict["fingertip_pos"] - obs_dict["fingertip_pos_rel_fixed"])
        return original(obs_dict, obs_order)

    factory_utils.collapse_obs_dict = collapse
    factory_utils._project_terms_installed = True
