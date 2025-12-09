# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

from cc.udp import UDP
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot import Humanoid
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite_lowlevel.policy.config import Cfg


# Load configuration
cfg = Cfg.from_arguments()

print(f"Policy frequency: {1 / cfg.policy_dt} Hz")

udp = UDP(("0.0.0.0", 11000), ("172.28.0.5", 11000))


# Initialize and start policy controller
controller = RlController(cfg)
controller.load_policy()

rate = RateLimiter(1 / cfg.policy_dt)

robot = Humanoid()

robot.enter_damping()

obs = robot.reset()

try:
    while True:
        actions = controller.update(obs)
        obs = robot.step(actions)
        udp.send_numpy(obs)

        rate.sleep()

except KeyboardInterrupt:
    robot.stop()

print("Stopped.")
