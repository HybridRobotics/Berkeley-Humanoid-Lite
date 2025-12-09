# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time
import struct
import scipy.spatial.transform as st

import serial

SYNC_1 = b'\x75'
SYNC_2 = b'\x65'


ser = serial.Serial("/dev/serial/by-path/pci-0000:00:14.0-usb-0:4.1:1.0", 1000000, timeout=0.001)

try:
    while True:
        t = time.perf_counter()
        
        sync_1 = ser.read(1)
        if not sync_1 or sync_1 != SYNC_1:
            # print("sync_1 error", sync_1)
            continue

        sync_2 = ser.read(1)
        if not sync_2 or sync_2 != SYNC_2:
            # print("sync_2 error", sync_2)
            continue

        size = ser.read(2)
        data_buffer = ser.read(28)

        data = struct.unpack("f"*7, data_buffer)

        # print([round(d, 2) for d in data])
        w, x, y, z, rx, ry, rz = data

        # roll pitch yaw
        euler = st.Rotation.from_quat([w, x, y, z], scalar_first=True).as_euler("xyz", degrees=True)
        
        print(f"x: {euler[0]:.2f} deg, y: {euler[1]:.2f} deg, z: {euler[2]:.2f} deg, freq: {1 / (time.perf_counter() - t):.2f} Hz", end="\r")

except KeyboardInterrupt:
    print()
    print("Keyboard interrupt")

ser.close()
