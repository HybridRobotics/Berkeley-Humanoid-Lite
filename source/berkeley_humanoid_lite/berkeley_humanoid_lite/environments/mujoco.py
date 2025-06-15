
import time
import threading

import numpy as np
import torch
import mujoco
import mujoco.viewer

from berkeley_humanoid_lite_lowlevel.policy.config import Cfg
from berkeley_humanoid_lite_lowlevel.policy.gamepad import Se2Gamepad


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q (torch.Tensor): Quaternion [w, x, y, z]
        v (torch.Tensor): Vector to rotate

    Returns:
        torch.Tensor: Rotated vector
    """
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (torch.dot(q_vec, v)) * 2.0
    return a - b + c


class MujocoEnv:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

        # Load appropriate MJCF model based on robot configuration
        if cfg.num_joints == 22:
            self.mj_model = mujoco.MjModel.from_xml_path("source/berkeley_humanoid_lite_assets/data/mjcf/bhl_scene.xml")
        else:
            self.mj_model = mujoco.MjModel.from_xml_path("source/berkeley_humanoid_lite_assets/data/mjcf/bhl_biped_scene.xml")

        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.cfg.physics_dt
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)


class MujocoVisualizer(MujocoEnv):
    """MuJoCo simulation environment for the Berkeley Humanoid Lite robot.

    This class handles the physics simulation, state observation, and control
    of the robot in the MuJoCo environment.

    Args:
        cfg (Cfg): Configuration object containing simulation parameters
    """
    def __init__(self, cfg: Cfg):
        super().__init__(cfg)

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


class MujocoSimulator(MujocoEnv):
    """MuJoCo simulation environment for the Berkeley Humanoid Lite robot.

    This class handles the physics simulation, state observation, and control
    of the robot in the MuJoCo environment.

    Args:
        cfg (Cfg): Configuration object containing simulation parameters
    """
    def __init__(self, cfg: Cfg):
        super().__init__(cfg)
        self.physics_substeps = int(np.round(self.cfg.policy_dt / self.cfg.physics_dt))

        # Initialize simulation parameters
        self.sensordata_dof_size = 3 * self.mj_model.nu
        self.gravity_vector = torch.tensor([0.0, 0.0, -1.0])

        # Initialize control parameters
        self.joint_kp = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)
        self.joint_kd = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)
        self.effort_limits = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)

        self.joint_kp[:] = torch.tensor(self.cfg.joint_kp)
        self.joint_kd[:] = torch.tensor(self.cfg.joint_kd)
        self.effort_limits[:] = torch.tensor(self.cfg.effort_limits)

        self.n_steps = 0

        print("Policy frequency: ", 1 / self.cfg.policy_dt)
        print("Physics frequency: ", 1 / self.cfg.physics_dt)
        print("Physics substeps: ", self.physics_substeps)

        # Initialize control mode and command variables
        self.is_killed = threading.Event()
        self.mode = 3.0  # Default to RL control mode
        self.command_velocity_x = 0.0
        self.command_velocity_y = 0.0
        self.command_velocity_yaw = 0.0

        # Start joystick thread
        self.command_controller = Se2Gamepad()
        self.command_controller.run()

    def reset(self) -> torch.Tensor:
        """Reset the simulation environment to initial state.

        Returns:
            torch.Tensor: Initial observations after reset
        """
        self.mj_data.qpos[0:3] = self.cfg.default_base_position
        self.mj_data.qpos[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Default quaternion orientation
        self.mj_data.qpos[7:] = self.cfg.default_joint_positions
        self.mj_data.qvel[:] = 0

        observations = self._get_observations()
        return observations

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Execute one simulation step with the given actions.

        Args:
            actions (torch.Tensor): Joint position targets for controlled joints

        Returns:
            torch.Tensor: Updated observations after executing the action
        """
        step_start_time = time.perf_counter()

        for _ in range(self.physics_substeps):
            self._apply_actions(actions)
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.mj_viewer.sync()
        observations = self._get_observations()

        # Maintain real-time simulation
        time_until_next_step = self.cfg.policy_dt - (time.perf_counter() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self.n_steps += 1
        return observations

    def _apply_actions(self, actions: torch.Tensor):
        """Apply control actions to the robot.

        Implements PD control with torque limits and filtering.

        Args:
            actions (torch.Tensor): Target joint positions for controlled joints
        """
        target_positions = torch.zeros((self.cfg.num_joints,))
        target_positions[self.cfg.action_indices] = actions

        # PD control
        output_torques = self.joint_kp * (target_positions - self._get_joint_pos()) + \
            self.joint_kd * (-self._get_joint_vel())

        # Apply EMA filtering and torque limits
        output_torques_clipped = torch.clip(output_torques, -self.effort_limits, self.effort_limits)

        self.mj_data.ctrl[:] = output_torques_clipped.numpy()

    def _get_base_pos(self) -> torch.Tensor:
        """Get base position of the robot.

        Returns:
            torch.Tensor: Base position [x, y, z]
        """
        return torch.tensor(self.mj_data.qpos[:3], dtype=torch.float32)

    def _get_base_quat(self) -> torch.Tensor:
        """Get base orientation quaternion from sensors.

        Returns:
            torch.Tensor: Base orientation quaternion [w, x, y, z]
        """
        return torch.tensor(self.mj_data.sensordata[self.sensordata_dof_size+0:self.sensordata_dof_size+4],
                          dtype=torch.float32)

    def _get_base_ang_vel(self) -> torch.Tensor:
        """Get base angular velocity from sensors.

        Returns:
            torch.Tensor: Base angular velocity [wx, wy, wz]
        """
        return torch.tensor(self.mj_data.sensordata[self.sensordata_dof_size+4:self.sensordata_dof_size+7],
                          dtype=torch.float32)

    def _get_projected_gravity(self) -> torch.Tensor:
        """Get gravity vector in the robot's base frame.

        Returns:
            torch.Tensor: Projected gravity vector
        """
        base_quat = self._get_base_quat()
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vector)
        return projected_gravity

    def _get_joint_pos(self) -> torch.Tensor:
        """Get joint positions from sensors.

        Returns:
            torch.Tensor: Joint positions
        """
        return torch.tensor(self.mj_data.sensordata[0:self.cfg.num_joints], dtype=torch.float32)

    def _get_joint_vel(self) -> torch.Tensor:
        """Get joint velocities from sensors.

        Returns:
            torch.Tensor: Joint velocities
        """
        return torch.tensor(self.mj_data.sensordata[self.cfg.num_joints:2*self.cfg.num_joints],
                          dtype=torch.float32)

    def _get_observations(self) -> torch.Tensor:
        """Get complete observation vector for the policy.

        Returns:
            torch.Tensor: Concatenated observation vector containing base orientation,
                         angular velocity, joint positions, velocities, and command state
        """
        command_mode_switch = self.command_controller.commands["mode_switch"]
        command_velocity_x = self.command_controller.commands["velocity_x"]
        command_velocity_y = self.command_controller.commands["velocity_y"]
        command_velocity_yaw = self.command_controller.commands["velocity_yaw"]

        if command_mode_switch != 0:
            self.mode = command_mode_switch
        self.command_velocity_x = command_velocity_x
        self.command_velocity_y = command_velocity_y * 0.5
        self.command_velocity_yaw = command_velocity_yaw

        return torch.cat([
            self._get_base_quat(),
            self._get_base_ang_vel(),
            self._get_joint_pos()[self.cfg.action_indices],
            self._get_joint_vel()[self.cfg.action_indices],
            torch.tensor([self.mode, self.command_velocity_x, self.command_velocity_y, self.command_velocity_yaw],
                        dtype=torch.float32),
        ], dim=-1)
