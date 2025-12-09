# Copyright (c) 2025, -T.K.-.

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel", help="CAN transport channel", type=str, default="can0")
    parser.add_argument("-i", "--id", help="CAN device ID", type=int, default=1)
    return parser.parse_args()
