# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import threading

import numpy as np
from cc.udp import UDP

from berkeley_humanoid_lite.environments import MujocoVisualizer, Cfg


if __name__ == "__main__":
    """Main execution function for the MuJoCo simulation environment."""
    # Initialize environment
    visualizer = MujocoVisualizer(Cfg(
        {
            "num_joints": 12,
            "physics_dt": 0.001,
        })
    )

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
