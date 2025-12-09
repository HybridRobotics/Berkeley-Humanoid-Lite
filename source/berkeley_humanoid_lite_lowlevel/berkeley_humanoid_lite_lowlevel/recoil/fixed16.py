# Copyright (c) 2025, -T.K.-.

import struct


class Fixed16:
    """
    A fixed-point 16-bit (Q8.8) number with 8 bits for the integer part and 8 bits for the fractional part.

    The range of the number is [-128.0, 127.9] with a resolution of 1/256.
    """
    def __init__(self, value: float):
        self.value = max(-128.0, min(127.9, value))

    def asFloat(self):
        return self.value

    def asBytes(self):
        int16_val = int(self.value * 256)
        return struct.pack("<h", int16_val)

    @staticmethod
    def fromBytes(data: bytes):
        int16_val, = struct.unpack("<h", data)
        return Fixed16(int16_val / 256.0)

    @staticmethod
    def fromInt(value: int):
        return Fixed16(value / 256.0)
