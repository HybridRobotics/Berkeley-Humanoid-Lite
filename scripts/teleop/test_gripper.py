import time
import struct

import serial


ser = serial.Serial("/dev/ttyUSB0", 115200)

while True:
    # 0.25: open
    # 0.85: closed
    # 0.9: tightly closed
    data = struct.pack("<ffb", 0.8, 0.8, 0x0C)
    ser.write(data)

    print(ser.readline())
