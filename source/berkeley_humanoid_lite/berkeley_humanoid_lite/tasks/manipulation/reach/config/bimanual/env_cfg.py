# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import berkeley_humanoid_lite.tasks.manipulation.reach.mdp as mdp
from berkeley_humanoid_lite.tasks.manipulation.reach.reach_env_cfg import ReachEnvCfg
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_BIMANUAL_CFG, HUMANOID_LITE_JOINTS

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


##
# Environment configuration
##

@configclass
class BimanualReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = HUMANOID_LITE_BIMANUAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["arm_left_elbow_roll"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["arm_left_elbow_roll"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["arm_left_elbow_roll"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "arm_left_shoulder_pitch_joint",
                "arm_left_shoulder_roll_joint",
                "arm_left_shoulder_yaw_joint",
                "arm_left_elbow_pitch_joint",
                "arm_left_elbow_roll_joint",
            ],
            scale=0.5,
            use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "arm_left_elbow_roll"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
