# Berkeley Humanoid Lite Low-level Control

This submodule repository contains the low-level control code for the Berkeley Humanoid Lite robot.


## Installation

```bash
sudo apt install net-tools can-utils
```

```bash
pip install -r requirements.txt
```

## Getting Started

### Bring up the CAN interface

The low-level computer connects to the joints via CAN.

Run this script to initialize the CAN interface:

```bash
sudo ./scripts/start_can_transports.sh
```

### Verify CAN connection

A Python script is provided to verify the CAN connection to all the joints of the robot:

```bash
python3 ./berkeley_humanoid_lite_lowlevel/robot/check_connection.py
```

### Launch the joystick receiver

The low-level computer also receives commands from the joystick.

To broadcast the joystick commands to other running nodes, run the following command:

```bash
python ./berkeley_humanoid_lite_lowlevel/policy/udp_joystick.py
```

### Joint Calibration

Because the joint actuators only have single encoder on the motor shaft, we need to calibrate the zero position of the joints after each power cycle.

Run the following command to start the calibration:

```bash
python3 ./berkeley_humanoid_lite_lowlevel/robot/calibrate_joints.py
```

After the script is launched and running, manually move the robot joints to the mechanical position limits. After all the joints are moved, press `q` or the `B` button on the joystick to quit the calibration.

The calibration data will be saved in the `./calibration.yaml` file.



### Run main controller

The main controller is implemented in C.

To run the controller, run the following command:

```bash
make run
```

Press `LB` + `A` to enter RL init mode. Then, press `RB` + `A` to enter RL running mode.

At any time, press `B` or the thumb buttons to exit RL mode. The joints will enter passive damping mode.

Press `Ctrl` + `C` to terminate the controller. Upon first termination, the joints will enter passive damping mode. Press `Ctrl` + `C` again to completely stop the controller, which joints will return to unpowered idle state.

