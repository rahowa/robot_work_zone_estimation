import math
import cv2
import numpy as np
from nptyping import Array
from typing import Sequence, Tuple


def compute_corner(marker_shape: Tuple[int, int], homography: Array[float]) -> Array[Tuple[int, int]]:
    h, w = marker_shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).astype(np.uint)


def draw_corner(frame: Array[int], corner: Array[Tuple[int, int]]) -> Array[int]:
    return cv2.polylines(frame, [np.int32(corner)], True, 255, 3, cv2.LINE_AA)


def projection_matrix(camera_parameters: Array[float], homography: Array[float]) -> Array[float]:
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def hex_to_rgb(hex_color: str) -> Sequence[int]:
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def mask_from_contours(contours: Sequence[Tuple[int, int]], image: Array[int]) -> Array[int]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for cnt in contours:
        mask = cv2.fillPoly(mask, [cnt], 255)
    return mask.astype(np.uint8)
