# Copyright (c) 2025, -T.K.-.

from .core import DataFrame


class CANFrame(DataFrame):
    ID_STANDARD = 0
    ID_EXTENDED = 1

    DEVICE_ID_MSK = 0x7F
    FUNC_ID_POS = 7
    FUNC_ID_MSK = 0x0F << FUNC_ID_POS

    def __init__(
        self,
        device_id: int = 0,
        func_id: int | None = None,
        size: int = 0,
        data: bytes = b""
    ):
        super().__init__(device_id, func_id, size, data)
        assert self.size <= 8
