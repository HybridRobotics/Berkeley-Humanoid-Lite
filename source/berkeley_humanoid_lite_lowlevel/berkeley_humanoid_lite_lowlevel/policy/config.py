import argparse

from omegaconf import DictConfig, ListConfig, OmegaConf


class Cfg(DictConfig):
    """Configuration class holding simulation and robot parameters."""

    # === Policy configurations ===
    policy_checkpoint_path: str

    # === Networking configurations ===
    ip_robot_addr: str
    ip_policy_obs_port: int
    ip_host_addr: str
    ip_policy_acs_port: int

    # === Physics configurations ===
    control_dt: float
    policy_dt: float
    physics_dt: float
    cutoff_freq: float

    # === Articulation configurations ===
    num_joints: int
    joints: list[str]
    joint_kp: list[float] | float
    joint_kd: list[float] | float
    effort_limits: list[float]
    default_base_position: list[float]
    default_joint_positions: list[float]

    # === Observation configurations ===
    num_observations: int
    history_length: int

    # === Command configurations ===
    command_velocity: list[float]

    # === Action configurations ===
    num_actions: int
    action_indices: list[int]
    action_scale: float
    action_limit_lower: float
    action_limit_upper: float

    @staticmethod
    def from_arguments() -> DictConfig | ListConfig:
        """
        Parse command line arguments and load configuration file.

        Returns:
            DictConfig | ListConfig: Loaded configuration object
        """
        parser = argparse.ArgumentParser(description="Policy Runner for Berkeley Humanoid Lite")
        parser.add_argument(
            "--config",
            type=str,
            default="./configs/policy_biped.yaml",
            help="Path to the configuration file",
        )
        args = parser.parse_args()

        print("Loading config file from ", args.config)

        with open(args.config, "r") as f:
            cfg = OmegaConf.load(f)

        return cfg
