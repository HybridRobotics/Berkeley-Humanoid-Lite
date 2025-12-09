# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time
import numpy as np

from loop_rate_limiters import RateLimiter
import berkeley_humanoid_lite_lowlevel.recoil as recoil


args = recoil.util.get_args()
bus = recoil.Bus(channel=args.channel, bitrate=1000000)

device_id = args.id

kp = 0.2
kd = 0.005

frequency = 1.0  # motion frequency is 1 Hz
amplitude = 1.0  # motion amplitude is 1 rad

rate = RateLimiter(frequency=200.0)


bus.write_position_kp(device_id, kp)
bus.write_position_kd(device_id, kd)
bus.write_torque_limit(device_id, 0.2)

bus.set_mode(device_id, recoil.Mode.POSITION)
bus.feed(device_id)

try:
    while True:
        target_angle = np.sin(2 * np.pi * frequency * time.time()) * amplitude

        measured_position, measured_velocity = bus.write_read_pdo_2(device_id, target_angle, 0.0)
        if measured_position is not None and measured_velocity is not None:
            print(f"Measured pos: {measured_position:.3f} \tvel: {measured_velocity:.3f}")

        rate.sleep()

except KeyboardInterrupt:
    pass

bus.set_mode(device_id, recoil.Mode.IDLE)
bus.stop()
