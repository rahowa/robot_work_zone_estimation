from configure import params
import os
import json
import typing as t
from glob import glob
from nptyping import Array
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
from tqdm import tqdm

CAM_WIDTH = 640
CAM_HEIGHT = 480


@dataclass
class CalibrationParams:
    cam_height: int
    cam_width: int
    chessboard_height: int
    chessboard_width: int
    criteria: t.Tuple[int, int, float]
    cam_shape: t.Tuple[int, int]
    chessboard_shape: t.Tuple[int, int]


@dataclass
class CameraParams:
    camera_mtx: Array[float]
    distortion_vec: Array[float]
    rotation_vec: Array[float]
    translation_vec: Array[float]


def collect_calibration_images(num_pictures: int,
                               save_path: str,
                               params: CalibrationParams) -> None:
    cap = cv2.VideoCapture(0)
    cap.set(3, params.cam_height)
    cap.set(4, params.cam_width)
    num_taken_pictures = 0

    while num_taken_pictures < num_pictures:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_founded, corners = cv2.findChessboardCorners(gray, params.chessboard_shape, None)
        if corners_founded:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), params.criteria)
            good_chess_img = cv2.drawChessboardCorners(gray, params.chessboard_shape, corners, ret)
            good_chess_img = cv2.putText(good_chess_img, "Good view",
                                         (params.cam_height//2, params.cam_width//2 - 30),
                                         cv2.FONT_HERSHEY_SIMPLEX, 2,
                                         (0, 255, 0))
            cv2.imwrite(os.path.join(save_path,
                        f'frame_{num_taken_pictures}.jpg'),
                        frame)
            num_taken_pictures += 1
            cv2.imshow("frame", good_chess_img)
        else:
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) % 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


def find_camera_params(path: str, params: CalibrationParams) -> CameraParams:
    default_obj_points = np.zeros((params.chessboard_height * params.chessboard_width, 3),
                                  np.float32)
    default_obj_points[:, :2] = np.mgrid[0:params.chessboard_width,
                                         0:params.chessboard_height].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    images = glob(path + '/*.jpg')
    for img_idx in tqdm(range(len(images))):
        img_path = images[img_idx]
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, params.chessboard_shape, None)
        if ret:
            obj_points.append(default_obj_points)
            img_points.append(corners)

    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                     img_points,
                                                     params.cam_shape,
                                                     None, None)
    return CameraParams(mtx, dist, rvecs, tvecs)


def print_camera_params(camera_params: CameraParams) -> None:
    print(("=" * 79).center(79))
    print(f"Camera matrix: {camera_params.camera_mtx}")
    print(("=" * 79).center(79))
    print(f"Distortion vector: {camera_params.distortion_vec}")
    print(("=" * 79).center(79))
    print(f"Rotation vector: {camera_params.rotation_vec}")
    print(("=" * 79).center(79))
    print(f"Translation vector: {camera_params.translation_vec}")
    print(("=" * 79).center(79))


def save_camera_params(params: CameraParams, path: str) -> None:
    params_dict = {
        "camera_mtx": params.camera_mtx.tolist(),
        "distortion_vec": params.distortion_vec.tolist(),
        "rotation_vec": params.rotation_vec.tolist(),
        "translation_vec": params.translation_vec.tolist()
    }
    with open(path, 'w') as save_file:
        json.dump(params_dict, save_file)


def load_camera_params(path: str) -> CameraParams:
    with open(path, "r") as params_file:
        params = json.load(params_file)
    return CameraParams(np.array(params['camera_mtx']),
                        np.array(params['distortion_vec']),
                        np.array(params['rotation_vec']),
                        np.array(params['translation_vec']))


# def undistort_frame()

def main(args: Namespace) -> None:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_params = CalibrationParams(CAM_HEIGHT,
                                           CAM_WIDTH,
                                           args.h,
                                           args.w,
                                           criteria,
                                           (CAM_HEIGHT, CAM_WIDTH),
                                           (args.h, args.w))
    if args.tp:
        collect_calibration_images(args.n, args.path, calibration_params)
    camera_params = find_camera_params(args.path, calibration_params)
    print_camera_params(camera_params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tp", action="store_true")
    parser.add_argument("--path", default='./data/calibration', type=str)
    parser.add_argument("--n", default=20, type=int)
    parser.add_argument("--w", default=6, type=int)
    parser.add_argument("--h", default=9, type=int)
    args = parser.parse_args()

    main(args)
