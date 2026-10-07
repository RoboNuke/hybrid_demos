"""The project's environments: import this module (inside an entry script's ``setup``)
to register them. Everything here imports Isaac Lab, so it must only be imported after
``AppLauncher`` has started the app -- which is exactly when ``setup`` runs."""

import gymnasium as gym

gym.register(
    # "forge" in the name is what lets the package's Forge wrappers (controller, fragile,
    # efficient reset, contact) apply -- require_forge_env checks the task NAME.
    id="Isaac-Forge-DemoPegInsert-Direct-v0",
    entry_point=f"{__name__}.forge_demo:ForgeDemoEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.forge_demo:ForgeDemoTaskCfg"},
)
