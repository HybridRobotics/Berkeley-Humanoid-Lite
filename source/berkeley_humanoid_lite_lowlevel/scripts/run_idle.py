# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import numpy as np
from cc.udp import UDP

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


robot = Humanoid()
# udp = UDP(send_addr=("172.28.0.5", 8000))

udp = UDP(("0.0.0.0", 11000), ("172.28.0.5", 11000))

try:
    while True:
        acs = np.zeros(12)
        obs = robot.step(acs)
        udp.send_numpy(obs)

        print(robot.joint_position_measured)

except KeyboardInterrupt:
    robot.stop()

print("Stopped.")
