# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
RL Controller

This module implements a policy runner that executes trained policies (PyTorch or ONNX)
for controlling the Berkeley Humanoid Lite robot. It handles UDP communication for
receiving observations and sending actions to the robot.
"""

from typing import Union
from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
import onnxruntime as ort


class Policy(ABC):
    """
    Abstract base class for all policies.
    """
    def __init__(self, checkpoint_path: str):
        pass

    @abstractmethod
    def forward(self, observations: np.ndarray) -> np.ndarray:
        pass


class TorchPolicy(Policy):
    """
    PyTorch policy inference runner.

    Loads and executes PyTorch models for robot control policies.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model: torch.nn.Module = torch.load(checkpoint_path, map_location=self.device)
        self.model.eval()

    def forward(self, observations: np.ndarray) -> np.ndarray:
        observations_tensor = torch.from_numpy(observations).unsqueeze(0).to(self.device)
        actions_tensor = self.model(observations_tensor)
        return actions_tensor.detach().cpu().squeeze(0).numpy()


class OnnxPolicy(Policy):
    """
    ONNX policy inference runner

    Loads and executes ONNX models for robot control policies.
    """
    def __init__(self, checkpoint_path: str):
        self.model: ort.InferenceSession = ort.InferenceSession(checkpoint_path)

        input_shape = self.model.get_inputs()[0].shape
        try:
            self.model.run(None, {"obs": np.zeros(input_shape, dtype=np.float32)})[0]
            self.key = "obs"
        except Exception as e:
            print(e)
            self.key = "onnx::Gemm_0"

    def forward(self, observations: np.ndarray) -> np.ndarray:
        return np.array(self.model.run(None, {self.key: observations})[0])


class RlController:
    """
    A class to run trained policies for the Berkeley Humanoid Lite robot.

    This class handles the execution of trained policies (PyTorch or ONNX format),
    processes robot observations, and sends control commands via UDP communication.
    """

    @staticmethod
    def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion [w, x, y, z]
            v (np.ndarray): Vector to rotate

        Returns:
            np.ndarray: Rotated vector
        """
        q_w = q[0]
        q_vec = q[1:4]
        a = v * (2.0 * q_w ** 2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * (np.dot(q_vec, v)) * 2.0
        return a - b + c

    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        """
        Initialize the PolicyRunner.

        Args:
            cfg (Union[DictConfig, ListConfig]): Configuration object containing policy parameters
        """
        self.cfg = cfg

        # for data logging
        self.data_log = []
        self.counter = 0

        # Initialize robot state buffers
        self.command_velocity = np.array(self.cfg.command_velocity, dtype=np.float32)

        if self.cfg.num_actions == self.cfg.num_joints:
            self.default_joint_positions = np.array(self.cfg.default_joint_positions, dtype=np.float32)
        else:
            self.default_joint_positions = np.array(self.cfg.default_joint_positions[10:], dtype=np.float32)

        self.gravity_vector = np.array([0., 0., -1.], dtype=np.float32)

        # Initialize observation and action buffers
        self.policy_observations = np.zeros((1, self.cfg.num_observations * (self.cfg.history_length + 1)), dtype=np.float32)
        self.policy_actions = np.zeros((1, self.cfg.num_actions), dtype=np.float32)
        self.prev_actions = np.zeros((self.cfg.num_actions,), dtype=np.float32)

    def load_policy(self) -> None:
        """
        Load the policy model (PyTorch or ONNX)
        """
        model_checkpoint_path = self.cfg.policy_checkpoint_path

        # Determine policy format and load appropriate model
        if ".pt" in model_checkpoint_path:
            torch.set_printoptions(precision=2)
            self.policy = TorchPolicy(model_checkpoint_path)
            print("Using Torch runner")

        elif ".onnx" in model_checkpoint_path:
            self.policy = OnnxPolicy(model_checkpoint_path)
            print("Using ONNX runner")

        else:
            raise ValueError("Unrecognized policy format")

    def update(self, robot_observations: np.ndarray) -> np.ndarray:
        """
        Run the policy execution loop.

        This method:
        1. Loads the policy model (PyTorch or ONNX)
        2. Sets up UDP communication
        3. Runs the main control loop
        4. Logs experiment data

        Args:
            robot_observations (np.ndarray): Observations from the robot low-level controller

        Returns:
            np.ndarray: Actions to send to the robot
        """

        # Parse UDP observations
        robot_base_quat = robot_observations[0:4]
        robot_base_ang_vel = robot_observations[4:7]
        robot_joint_pos = robot_observations[7:7 + self.cfg.num_actions] - self.default_joint_positions
        robot_joint_vel = robot_observations[7 + self.cfg.num_actions:7 + self.cfg.num_actions * 2]
        # robot_mode = robot_observations[7 + self.cfg.num_actions * 2]
        command_velocity = robot_observations[7 + self.cfg.num_actions * 2 + 1:7 + self.cfg.num_actions * 2 + 4]

        # Process observations
        base_ang_vel = robot_base_ang_vel
        projected_gravity = self.quat_rotate_inverse(robot_base_quat, self.gravity_vector)
        joint_pos = robot_joint_pos
        joint_vel = robot_joint_vel

        # Update observation buffer
        self.policy_observations[:] = np.concatenate([
            self.policy_observations[0, self.cfg.num_observations:],
            command_velocity,
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
            self.prev_actions
        ], axis=0)

        # Execute policy
        self.policy_actions[:] = self.policy.forward(self.policy_observations)

        # Process and scale actions
        policy_actions_clipped = np.clip(self.policy_actions[0],
                                         self.cfg.action_limit_lower,
                                         self.cfg.action_limit_upper)
        self.prev_actions[:] = policy_actions_clipped

        policy_actions_scaled = policy_actions_clipped * self.cfg.action_scale + self.default_joint_positions

        # # Log data
        # data_log.append(np.concatenate([[time.time()], policy_observations.flatten()]).tolist())
        # counter += 1

        # # Save experiment data
        # with open("data_log.json", "w") as f:
        #     json.dump(data_log, f)
        # print("Written experiment data to ./data_log.json")

        return policy_actions_scaled
