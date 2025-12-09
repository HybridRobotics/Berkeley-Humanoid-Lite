
import argparse
from typing import Union

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController


def parse_arguments() -> Union[DictConfig, ListConfig]:
    """
    Parse command line arguments and load configuration file.

    Returns:
        Union[DictConfig, ListConfig]: Loaded configuration object
    """
    parser = argparse.ArgumentParser(description="Policy Runner for Berkeley Humanoid Lite")
    parser.add_argument("--config", type=str, default="./configs/policy_humanoid.yaml",
                       help="Path to the configuration file")
    args = parser.parse_args()

    print("Loading config file from ", args.config)

    with open(args.config, "r") as f:
        cfg = OmegaConf.load(f)

    return cfg


# Load configuration
cfg = parse_arguments()


# Initialize and start policy controller
controller = RlController(cfg)
controller.load_policy()

obs = np.zeros((63,), dtype=np.float32)  # Example observation, adjust as needed

actions = controller.update(obs)