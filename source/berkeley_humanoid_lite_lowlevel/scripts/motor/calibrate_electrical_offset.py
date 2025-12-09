# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil


args = recoil.util.get_args()
bus = recoil.Bus(channel=args.channel, bitrate=1000000)

device_id = args.id

bus.set_mode(device_id, recoil.Mode.CALIBRATION)

# the motor should now perform the calibration sequence

# wait for calibration to finish
time.sleep(20)

bus.stop()
