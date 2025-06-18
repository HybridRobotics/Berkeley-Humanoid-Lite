# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot.bimanual import Bimanual


np.set_printoptions(precision=3, suppress=True)

rate = RateLimiter(100)

robot = Bimanual()

robot.run(kp=20, kd=2, torque_limit=0.5)

try:
    while True:
        obs = robot.step(np.zeros((10,), dtype=np.float32))

        print(obs)

        rate.sleep()

except KeyboardInterrupt:
    print("Stopping robot due to keyboard interrupt.")

robot.stop()
