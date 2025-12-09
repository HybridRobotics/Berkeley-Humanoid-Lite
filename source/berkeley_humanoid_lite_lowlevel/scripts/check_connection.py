# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


robot = Humanoid()

robot.check_connection()

robot.stop()
