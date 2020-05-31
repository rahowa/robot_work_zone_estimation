import sys
import cv2
from argparse import ArgumentParser, Namespace

sys.path.append('./src')

from src.calibrate_camera_utils import (CalibrationParams, collect_calibration_images,
                                        find_camera_params, print_camera_params)


def main_calibration(args: Namespace) -> None:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_params = CalibrationParams(args.cam_height,
                                           args.cam_width,
                                           args.y,
                                           args.x,
                                           criteria,
                                           (args.cam_height, args.cam_width),
                                           (args.y, args.x))
    if args.t:
        collect_calibration_images(args.n, args.p, calibration_params)
    camera_params = find_camera_params(args.path, calibration_params)
    print_camera_params(camera_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--take_pictures", "-t",
                        help="Allow to take valid pictures for calibration from web-cam",
                        action="store_true")
    parser.add_argument("--calibration_path", "-p",
                        help="Path to images with chessboard for calibration",
                        default='./data/calibration', type=str)
    parser.add_argument("--num_samples", "-n",
                        help="Number of pictures for calibration",
                        default=20, type=int)
    parser.add_argument("--horizontal_blocks", "-x",
                        help="Number of horizontal squares in chessboard (x axis)",
                        default=6, type=int)
    parser.add_argument("--vertical_blocks", "-y",
                        help="Number of vertical squares in chessboard (y axis)",
                        default=9, type=int)
    parser.add_argument("--cam_height", "--m",
                        help="Camera height in pixels",
                        default=480, type=int)
    parser.add_argument("--cam_width", "--w",
                        help="Camera width in pixels",
                        default=640, type=int)
    args = parser.parse_args()

    main_calibration(args)
