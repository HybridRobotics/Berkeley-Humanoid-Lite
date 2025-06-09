
import gymnasium as gym

from . import agents, env_cfg


##
# Register Gym environments.
##

gym.register(
    id="Reach-Berkeley-Humanoid-Lite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.BimanualReachEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.BimanualReachPPORunnerCfg,
    },
)
