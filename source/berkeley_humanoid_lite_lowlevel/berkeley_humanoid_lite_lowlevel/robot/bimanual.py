# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import struct
import time

import serial
import numpy as np

import berkeley_humanoid_lite_lowlevel.recoil as recoil


class Bimanual:
    def __init__(self):

        self.left_arm_transport = recoil.Bus("can0")
        self.right_arm_transport = recoil.Bus("can1")
        self.gripper = serial.Serial("/dev/ttyUSB0", 115200)

        self.joints = [
            (self.left_arm_transport, 1, "left_shoulder_pitch_joint"),
            (self.left_arm_transport, 3, "left_shoulder_roll_joint"),
            (self.left_arm_transport, 5, "left_shoulder_yaw_joint"),
            (self.left_arm_transport, 7, "left_elbow_joint"),
            (self.left_arm_transport, 9, "left_wrist_yaw_joint"),

            (self.right_arm_transport, 2, "right_shoulder_pitch_joint"),
            (self.right_arm_transport, 4, "right_shoulder_roll_joint"),
            (self.right_arm_transport, 6, "right_shoulder_yaw_joint"),
            (self.right_arm_transport, 8, "right_elbow_joint"),
            (self.right_arm_transport, 10, "right_wrist_yaw_joint"),
        ]

        self.joint_axis_directions = np.array([
            +1, +1, -1, -1, -1,
            -1, +1, -1, +1, -1,
            +1, +1,  # gripper
        ], dtype=np.float32)

        self.position_offsets = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.2, 0.2,  # gripper
        ], dtype=np.float32)

        self.joint_position_target = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_position_measured = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_velocity_measured = np.zeros(len(self.joints), dtype=np.float32)
        self.gripper_left_target = 0.5
        self.gripper_right_target = 0.5

    def start(self, kp=20, kd=2, torque_limit=1):
        self.joint_kp = np.zeros((len(self.joints),), dtype=np.float32)
        self.joint_kd = np.zeros((len(self.joints),), dtype=np.float32)
        self.torque_limit = np.zeros((len(self.joints),), dtype=np.float32)

        self.joint_kp[:] = kp
        self.joint_kd[:] = kd
        self.torque_limit[:] = torque_limit

        for i, entry in enumerate(self.joints):
            bus, device_id, joint_name = entry

            print(f"Initializing joint {joint_name}:")
            print(f"  kp: {self.joint_kp[i]}, kd: {self.joint_kd[i]}, torque limit: {self.torque_limit[i]}")

            # Set the joint mode to idle
            bus.set_mode(device_id, recoil.Mode.IDLE)
            time.sleep(0.001)
            bus.write_position_kp(device_id, self.joint_kp[i])
            time.sleep(0.001)
            bus.write_position_kd(device_id, self.joint_kd[i])
            time.sleep(0.001)
            bus.write_torque_limit(device_id, self.torque_limit[i])
            time.sleep(0.001)
            bus.feed(device_id)
            bus.set_mode(device_id, recoil.Mode.POSITION)

            self.position_offsets[i] = bus.read_position_measured(device_id) * self.joint_axis_directions[i]

        print("Motors enabled")
        print(self.position_offsets)

    def stop(self):
        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.DAMPING)

        print("Entered damping mode. Press Ctrl+C again to exit.\n")

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Exiting damping mode.")

        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.IDLE)

        self.left_arm_transport.stop()
        self.right_arm_transport.stop()

    def get_observations(self) -> np.ndarray:
        return np.concatenate([
            self.joint_position_measured[:],
            [self.gripper_left_target, self.gripper_right_target],
        ])

    def update_joint_group(self, joint_id_l, joint_id_r):
        # adjust direction and offset of target values
        position_target_l = (self.joint_position_target[joint_id_l] + self.position_offsets[joint_id_l]) * self.joint_axis_directions[joint_id_l]
        position_target_r = (self.joint_position_target[joint_id_r] + self.position_offsets[joint_id_r]) * self.joint_axis_directions[joint_id_r]

        self.joints[joint_id_l][0].transmit_pdo_2(self.joints[joint_id_l][1], position_target=position_target_l, velocity_target=0.0)
        self.joints[joint_id_r][0].transmit_pdo_2(self.joints[joint_id_r][1], position_target=position_target_r, velocity_target=0.0)

        position_measured_l, velocity_measured_l = self.joints[joint_id_l][0].receive_pdo_2(self.joints[joint_id_l][1])
        position_measured_r, velocity_measured_r = self.joints[joint_id_r][0].receive_pdo_2(self.joints[joint_id_r][1])

        # adjust direction and offset of target values
        if position_measured_l is not None:
            self.joint_position_measured[joint_id_l] = (position_measured_l * self.joint_axis_directions[joint_id_l]) - self.position_offsets[joint_id_l]
        if velocity_measured_l is not None:
            self.joint_velocity_measured[joint_id_l] = velocity_measured_l * self.joint_axis_directions[joint_id_l]
        if position_measured_r is not None:
            self.joint_position_measured[joint_id_r] = (position_measured_r * self.joint_axis_directions[joint_id_r]) - self.position_offsets[joint_id_r]
        if velocity_measured_r is not None:
            self.joint_velocity_measured[joint_id_r] = velocity_measured_r * self.joint_axis_directions[joint_id_r]

    def update_joints(self):

        # communicate with actuators
        self.update_joint_group(0, 5)
        self.update_joint_group(1, 6)
        self.update_joint_group(2, 7)
        self.update_joint_group(3, 8)
        self.update_joint_group(4, 9)

        # communicate with gripper
        # 0.2: open
        # 0.85: closed
        # 0.9: tightly closed
        gripper_left_raw_value = 0.2 + self.gripper_left_target * 0.6
        gripper_right_raw_value = 0.2 + self.gripper_right_target * 0.6
        data = struct.pack("<ffb", gripper_left_raw_value, gripper_right_raw_value, 0x0C)
        self.gripper.write(data)

    def reset(self):
        obs = self.get_observations()
        return obs

    def step(self, actions: np.ndarray):
        """
        actions: np.ndarray of shape (n_joints, )
        """
        self.joint_position_target[:] = actions[0:10]
        self.gripper_left_target = actions[10]
        self.gripper_right_target = actions[11]

        self.update_joints()

        obs = self.get_observations()

        return obs

    def check_connection(self):
        for entry in self.joints:
            bus, device_id, joint_name = entry
            print(f"Pinging {joint_name} ... ", end="\t")
            result = bus.ping(device_id)
            if result:
                print("OK")
            else:
                print("ERROR")
            time.sleep(0.1)
