# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import time
import struct
import threading

import numpy as np
from loop_rate_limiters import RateLimiter
import serial


class ImuRegisters:
    """
    Register address mapping for the IMU.
    """
    SAVE                    = 0x00  # save/reboot/factory reset
    CALSW                   = 0x01  # calibration mode
    RSW                     = 0x02  # output content
    RRATE                   = 0x03  # output rate
    BAUD                    = 0x04  # serial port baudrate
    AXOFFSET                = 0x05  # acceleration X bias
    AYOFFSET                = 0x06  # acceleration Y bias
    AZOFFSET                = 0x07  # acceleration Z bias
    GXOFFSET                = 0x08  # angular velocity X bias
    GYOFFSET                = 0x09  # angular velocity Y bias
    GZOFFSET                = 0x0A  # angular velocity Z bias
    HXOFFSET                = 0x0B  # magnetic field X bias
    HYOFFSET                = 0x0C  # magnetic field Y bias
    HZOFFSET                = 0x0D  # magnetic field Z bias
    D0MODE                  = 0x0E  # digital pin 0 mode
    D1MODE                  = 0x0F  # digital pin 1 mode
    D2MODE                  = 0x10  # digital pin 2 mode
    D3MODE                  = 0x11  # digital pin 3 mode
    IICADDRESS              = 0x1A  # I2C device address
    LEDOFF                  = 0x1B  # turn off the LED light
    MAGRANGX                = 0x1C  # magnetic field X calibration range
    MAGRANGY                = 0x1D  # magnetic field Y calibration range
    MAGRANGZ                = 0x1E  # magnetic field Z calibration range
    BANDWIDTH               = 0x1F  # bandwidth
    GYRORANGE               = 0x20  # gyroscope range
    ACCRANGE                = 0x21  # acceleration range
    SLEEP                   = 0x22  # hibernate
    ORIENT                  = 0x23  # installation direction
    AXIS6                   = 0x24  # algorithm selection
    FILTK                   = 0x25  # dynamic filtering
    GPSBAUD                 = 0x26  # GPS baudrate
    READADDR                = 0x27  # read register
    ACCFILT                 = 0x2A  # acceleration filter
    POWONSEND               = 0x2D  # command start
    VERSION                 = 0x2E  # version number
    YYMM                    = 0x30  # year and month
    DDHH                    = 0x31  # day and hour
    MMSS                    = 0x32  # minute and second
    MS                      = 0x33  # millisecond
    AX                      = 0x34  # acceleration X
    AY                      = 0x35  # acceleration Y
    AZ                      = 0x36  # acceleration Z
    GX                      = 0x37  # angular velocity X
    GY                      = 0x38  # angular velocity Y
    GZ                      = 0x39  # angular velocity Z
    HX                      = 0x3A  # magnetic field X
    HY                      = 0x3B  # magnetic field Y
    HZ                      = 0x3C  # magnetic field Z
    ROLL                    = 0x3D  # roll angle
    PITCH                   = 0x3E  # pitch angle
    YAW                     = 0x3F  # yaw angle
    TEMP                    = 0x40  # temperature
    D0STATUS                = 0x41  # digital pin 0 state
    D1STATUS                = 0x42  # digital pin 1 state
    D2STATUS                = 0x43  # digital pin 2 state
    D3STATUS                = 0x44  # digital pin 3 state
    PRESSUREL               = 0x45  # pressure lower 16 bits
    PRESSUREH               = 0x46  # pressure upper 16 bits
    HEIGHTL                 = 0x47  # height lower 16 bits
    HEIGHTH                 = 0x48  # height upper 16 bits
    LONL                    = 0x49  # longitude lower 16 bits
    LONH                    = 0x4A  # longitude upper 16 bits
    LATL                    = 0x4B  # latitude lower 16 bits
    LATH                    = 0x4C  # latitude upper 16 bits
    GPSHEIGHT               = 0x4D  # GPS altitude
    GPSYAW                  = 0x4E  # GPS heading angle
    GPSVL                   = 0x4F  # GPS ground speed lower 16 bits
    GPSVH                   = 0x50  # GPS ground speed upper 16 bits
    Q0                      = 0x51  # quaternion 0
    Q1                      = 0x52  # quaternion 1
    Q2                      = 0x53  # quaternion 2
    Q3                      = 0x54  # quaternion 3
    SVNUM                   = 0x55  # number of visible satellites
    PDOP                    = 0x56  # position accuracy
    HDOP                    = 0x57  # horizontal accuracy
    VDOP                    = 0x58  # vertical accuracy
    DELAYT                  = 0x59  # alarm signal delay
    XMIN                    = 0x5A  # X-axis angle alarm minimum value
    XMAX                    = 0x5B  # X-axis angle alarm maximum value
    BATVAL                  = 0x5C  # battery supply voltage
    ALARMPIN                = 0x5D  # alarm pin mapping
    YMIN                    = 0x5E  # Y-axis angle alarm minimum value
    YMAX                    = 0x5F  # Y-axis angle alarm maximum value
    GYROCALITHR             = 0x61  # gyroscope still threshold
    ALARMLEVEL              = 0x62  # alarm level
    GYROCALTIME             = 0x63  # gyroscope auto calibration time
    TRIGTIME                = 0x68  # alarm continuous trigger time
    KEY                     = 0x69  # unlock key
    WERROR                  = 0x6A  # gyroscope change value
    TIMEZONE                = 0x6B  # GPS time zone
    WZTIME                  = 0x6E  # angular velocity continuous rest time
    WZSTATIC                = 0x6F  # angular velocity integral threshold
    MODDELAY                = 0x74  # RS485 data response delay
    XREFROLL                = 0x79  # roll angle zero reference value
    YREFPITCH               = 0x7A  # pitch angle zero reference value
    NUMBERID1               = 0x7F  # number ID 1-2
    NUMBERID2               = 0x80  # number ID 3-4
    NUMBERID3               = 0x81  # number ID 5-6
    NUMBERID4               = 0x82  # number ID 7-8
    NUMBERID5               = 0x83  # number ID 9-10
    NUMBERID6               = 0x84  # number ID 11-12


class FrameType:
    TIME                    = 0x50
    ACCELERATION            = 0x51
    ANGULAR_VELOCITY        = 0x52
    ANGLE                   = 0x53
    MAGNETIC_FIELD          = 0x54
    PORT_STATUS             = 0x55
    BAROMETER_ALTITUDE      = 0x56
    LATITUDE_LONGITUDE      = 0x57
    GROUND_SPEED            = 0x58
    QUATERNION              = 0x59
    GPS_POSITION_ACCURACY   = 0x5A
    READY                   = 0x5F


class SamplingRate:
    RATE_0_2_HZ     = 0x01
    RATE_0_5_HZ     = 0x02
    RATE_1_HZ       = 0x03
    RATE_2_HZ       = 0x04
    RATE_5_HZ       = 0x05
    RATE_10_HZ      = 0x06
    RATE_20_HZ      = 0x07
    RATE_50_HZ      = 0x08
    RATE_100_HZ     = 0x09
    RATE_200_HZ     = 0x0B
    RATE_SINGLE     = 0x0C
    RATE_NO_RETURN  = 0x0D


class Baudrate:
    BAUD_4800       = 0x01
    BAUD_9600       = 0x02
    BAUD_19200      = 0x03
    BAUD_38400      = 0x04
    BAUD_57600      = 0x05
    BAUD_115200     = 0x06
    BAUD_230400     = 0x07
    BAUD_460800     = 0x08
    # BAUD_921600     = 0x09


class SerialImu:
    """
    Driver for the HiWonder IM10A 10-axis USB IMU.

    @see https://www.hiwonder.com/products/imu-module

    To change configurations, the following steps can be performed first:
    1. send UNLOCK command
    2. send commands to modify or read configuration data
    3. send SAVE command to apply the settings

    The commands must be completed within 10 seconds, otherwise the IMU will
    automatically be locked.
    """
    FRAME_LENGTH = 11

    @staticmethod
    def baud_to_int(baudrate: int) -> int:
        match baudrate:
            case Baudrate.BAUD_4800:
                return 4800
            case Baudrate.BAUD_9600:
                return 9600
            case Baudrate.BAUD_19200:
                return 19200
            case Baudrate.BAUD_38400:
                return 38400
            case Baudrate.BAUD_57600:
                return 57600
            case Baudrate.BAUD_115200:
                return 115200
            case Baudrate.BAUD_230400:
                return 230400
            case Baudrate.BAUD_460800:
                return 460800
            # case Baudrate.BAUD_921600:
            #     return 921600
        return 0

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = Baudrate.BAUD_115200, read_timeout=4):
        self.port: str = port
        self.baud: int = baudrate
        self.read_timeout: int = read_timeout

        baudrate_int = self.baud_to_int(baudrate)
        self.ser: serial.Serial = serial.Serial(self.port, baudrate_int, timeout=self.read_timeout)

        print("Serial is Opened:", self.ser.is_open)

        self.is_stopped: threading.Event = threading.Event()
        self.is_stopped.clear()

        # === IMU readings ===
        # seconds
        self.timestamp: float = 0.0
        # Celcius degree
        self.temperature: float = 0.0
        # (x, y, z) m/s^2
        self.acceleration: np.ndarray = np.zeros(3, dtype=np.float32)
        # (x, y, z) deg/s
        self.angular_velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        # (yaw, pitch, roll) deg
        self.angle: np.ndarray = np.zeros(3, dtype=np.float32)
        # (x, y, z) μT
        self.magnetic_field: np.ndarray = np.zeros(3, dtype=np.float32)
        # (w, x, y, z)
        self.quaternion: np.ndarray = np.zeros(4, dtype=np.float32)

        # self.__debug_last_time: float = time.perf_counter_ns()

    def __read_frame(self) -> None:
        """
        Parse a frame from the serial port.
        """
        start = self.ser.read(1)
        start, = struct.unpack("<B", start)
        if start != 0x55:
            return
        frame = self.ser.read(self.FRAME_LENGTH - 1)

        frame_type, data1, data2, data3, data4, sumcrc = struct.unpack("<BhhhhB", frame)

        # print(type, data1, data2, data3, data4, sumcrc)
        if frame_type == FrameType.TIME:
            _, year, month, day, hour, minute, second, millisecond = struct.unpack("<BBBBhB", frame)

        elif frame_type == FrameType.ACCELERATION:
            self.acceleration[0] = data1 * 16.0 / 32768.0  # g
            self.acceleration[1] = data2 * 16.0 / 32768.0  # g
            self.acceleration[2] = data3 * 16.0 / 32768.0  # g
            self.temperature = data4 / 100.0  # Celsius

        elif frame_type == FrameType.ANGULAR_VELOCITY:
            self.angular_velocity[0] = data1 * 2000.0 / 32768.0  # deg/s
            self.angular_velocity[1] = data2 * 2000.0 / 32768.0  # deg/s
            self.angular_velocity[2] = data3 * 2000.0 / 32768.0  # deg/s

            # for debugging
            # print(f"frequency: {1.0 / ((time.perf_counter_ns() - self.__debug_last_time) / 1e9)} Hz")
            # self.__debug_last_time = time.perf_counter_ns()

        elif frame_type == FrameType.ANGLE:
            self.angle[0] = data1 * 180.0 / 32768.0  # deg
            self.angle[1] = data2 * 180.0 / 32768.0  # deg
            self.angle[2] = data3 * 180.0 / 32768.0  # deg

        elif frame_type == FrameType.MAGNETIC_FIELD:
            self.magnetic_field[0] = data1 * 1.0 / 32768.0
            self.magnetic_field[1] = data2 * 1.0 / 32768.0
            self.magnetic_field[2] = data3 * 1.0 / 32768.0

        elif frame_type == FrameType.QUATERNION:
            self.quaternion[0] = data1 * 1.0 / 32768.0
            self.quaternion[1] = data2 * 1.0 / 32768.0
            self.quaternion[2] = data3 * 1.0 / 32768.0
            self.quaternion[3] = data4 * 1.0 / 32768.0

    def run(self) -> None:
        """
        Start the IMU reading loop.
        """
        self.start_time = time.time()
        while not self.is_stopped.is_set():
            self.__read_frame()

    def run_forever(self) -> None:
        """
        Start the IMU reading loop in a separate thread with high priority.
        """
        # self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

        # Set thread priority to high
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            process.nice(0)  # Set process priority to high (-20 is highest, 19 is lowest)
        except (ImportError, PermissionError):
            print("Warning: Could not set thread priority. Running with default priority.")

    def stop(self) -> None:
        """
        Stop the IMU reading loop.
        """
        self.is_stopped.set()

    def unlock(self) -> None:
        """
        Unlock the IMU to allow configuration changes.
        """
        register_addr = 0x69
        data = 0xB588
        frame = struct.pack("<BBBH", 0xFF, 0xAA, register_addr, data)
        self.ser.write(frame)

    def save(self) -> None:
        """
        Save the configurations permanently.
        """
        register_addr = 0x00
        data = 0x0000
        frame = struct.pack("<BBBH", 0xFF, 0xAA, register_addr, data)
        self.ser.write(frame)

    def write_frame(self, register_addr: int, data: int) -> None:
        frame = struct.pack("<BBBH", 0xFF, 0xAA, register_addr, data)
        self.ser.write(frame)

    def set_output_content(
        self,
        time: bool = False,
        acceleration: bool = False,
        angular_velocity: bool = False,
        angle: bool = False,
        magnetic_field: bool = False,
        port_status: bool = False,
        pressure: bool = False,
        gps: bool = False,
        velocity: bool = False,
        quaternion: bool = False,
        gps_position_accuracy: bool = False,
    ) -> None:
        """Set the output content configuration for the IMU.

        Args:
            time: Enable time output
            acceleration: Enable acceleration output
            angular_velocity: Enable angular velocity output
            angle: Enable angle output
            magnetic_field: Enable magnetic field output
            port_status: Enable port status output
            pressure: Enable pressure output
            gps: Enable GPS output
            velocity: Enable velocity output
            quaternion: Enable quaternion output
            gps_position_accuracy: Enable GPS position accuracy output

        Returns:
            int: The output content configuration value where each bit represents
                 whether that particular data type should be included in the output.
        """
        # Validate all inputs are boolean
        for param_name, param_value in locals().items():
            if param_name != "self" and not isinstance(param_value, bool):
                raise TypeError(f"Parameter {param_name} must be a boolean")

        output_content = 0x00
        output_content |= (time << 0)
        output_content |= (acceleration << 1)
        output_content |= (angular_velocity << 2)
        output_content |= (angle << 3)
        output_content |= (magnetic_field << 4)
        output_content |= (port_status << 5)
        output_content |= (pressure << 6)
        output_content |= (gps << 7)
        output_content |= (velocity << 8)
        output_content |= (quaternion << 9)
        output_content |= (gps_position_accuracy << 10)

        self.write_frame(ImuRegisters.RSW, output_content)

    def set_sampling_rate(self, rate: int) -> None:
        self.write_frame(ImuRegisters.RRATE, rate)

    def set_baudrate(self, baudrate: int) -> None:
        self.baud = baudrate

        # wait for the unlock to be set
        time.sleep(0.1)
        self.write_frame(ImuRegisters.BAUD, baudrate)
        # wait for the baudrate to be set
        time.sleep(0.1)

        # reopen the serial port with the new baudrate
        del self.ser
        baudrate_int = self.baud_to_int(baudrate)
        self.ser = serial.Serial(self.port, baudrate_int, timeout=self.read_timeout)


if __name__ == "__main__":
    imu = SerialImu(baudrate=Baudrate.BAUD_460800)

    # change baudrate:
    # imu.unlock()
    # time.sleep(0.1)
    # imu.set_baudrate(Baudrate.BAUD_115200)
    # time.sleep(0.1)
    # imu.save()

    # set sampling rate:
    # imu.unlock()
    # imu.set_sampling_rate(SamplingRate.RATE_200_HZ)
    # imu.save()

    # # set output content:
    # imu.unlock()
    # time.sleep(0.1)
    # imu.set_output_content(
    #     acceleration=True,
    #     angular_velocity=True,
    #     quaternion=True,
    # )
    # time.sleep(0.1)
    # imu.save()

    # time.sleep(0.1)
    # imu_reader.save()
    # # time.sleep(0.2)

    imu.run_forever()

    rate = RateLimiter(100)

    print("IMU reader started")

    try:
        while True:
            print(f"ax: {imu.acceleration[0]:.2f}\tay: {imu.acceleration[1]:.2f}\taz: {imu.acceleration[2]:.2f}", end="\t")
            print(f"gx: {imu.angular_velocity[0]:.2f}\tgy: {imu.angular_velocity[1]:.2f}\tgz: {imu.angular_velocity[2]:.2f}", end="\t")
            print(f"qw: {imu.quaternion[0]:.2f}\tqx: {imu.quaternion[1]:.2f}\tqy: {imu.quaternion[2]:.2f}\tqz: {imu.quaternion[3]:.2f}")
            rate.sleep()
    except KeyboardInterrupt:
        imu.stop()
        imu.ser.close()
        print("IMU reader stopped")
