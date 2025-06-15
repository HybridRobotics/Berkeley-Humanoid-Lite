# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import threading

import numpy as np
import torch
import mujoco
import mujoco.viewer
from cc.udp import UDP


class MujocoVisualizer:
    """MuJoCo simulation environment for the Berkeley Humanoid Lite robot.

    This class handles the physics simulation, state observation, and control
    of the robot in the MuJoCo environment.

    Args:
        cfg (Cfg): Configuration object containing simulation parameters
    """
    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_path("source/berkeley_humanoid_lite_assets/data/mjcf/bhl_biped_scene.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.001  # Set physics timestep
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.num_dofs = self.mj_model.nu
        print(f"Number of DOFs: {self.num_dofs}")

    def reset(self) -> None:
        """Reset the simulation environment to initial state.

        Returns:
            torch.Tensor: Initial observations after reset
        """
        self.mj_data.qpos[0:3] = np.array([0.0, 0.0, 0.0])  # Reset base position to origin
        self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion orientation
        self.mj_data.qpos[7:7 + self.num_dofs] = 0
        self.mj_data.qvel[:] = 0

    def step(self, robot_observations: np.array) -> None:
        """Execute one simulation step with the given actions.

        Args:
            actions (torch.Tensor): Joint position targets for controlled joints

        Returns:
            torch.Tensor: Updated observations after executing the action
        """
        robot_base_quat = robot_observations[0:4]
        robot_base_ang_vel = robot_observations[4:7]
        robot_joint_pos = robot_observations[7:7 + self.num_dofs]
        robot_joint_vel = robot_observations[7 + self.num_dofs:7 + self.num_dofs * 2]
        robot_mode = robot_observations[7 + self.num_dofs * 2]
        command_velocity = robot_observations[7 + self.num_dofs * 2 + 1:7 + self.num_dofs * 2 + 4]

        self.mj_data.qpos[0:3] = np.array([0.0, 0.0, 0.0])
        self.mj_data.qpos[3:7] = robot_base_quat
        self.mj_data.qvel[0:3] = np.array([0.0, 0.0, 0.0])
        self.mj_data.qvel[3:6] = robot_base_ang_vel
        self.mj_data.qpos[7:] = robot_joint_pos
        self.mj_data.qvel[6:] = robot_joint_vel

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_viewer.sync()


if __name__ == "__main__":
    """Main execution function for the MuJoCo simulation environment."""
    # Initialize environment
    visualizer = MujocoVisualizer()

    def receive_udp_data(robot_observation_buffer):
        # Setup UDP communication
        udp = UDP(("0.0.0.0", 11000), ("127.0.0.1", 11000))

        """Thread function to receive UDP data."""
        while True:
            robot_observations = udp.recv_numpy(dtype=np.float32)
            if robot_observations is not None:
                robot_observation_buffer[:] = robot_observations

    robot_observation_buffer = np.zeros((35,), dtype=np.float32)

    udp_receive_thread = threading.Thread(target=receive_udp_data, args=(robot_observation_buffer,))
    udp_receive_thread.daemon = True
    udp_receive_thread.start()

    while True:
        visualizer.step(robot_observation_buffer)
