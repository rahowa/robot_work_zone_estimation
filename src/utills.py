import math
import cv2
import numpy as np
from nptyping import Array
from typing import List, Sequence, Union, Tuple
# from src.kalman import KalmanFilter
from .obj_loader import OBJ


# def compute_corner(marker_shape: Tuple[int, int], homography: Array[float], filter: KalmanFilter = None) -> Array[Tuple[int, int], 4]:
#     h, w = marker_shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#
#     if filter is None:
#         return cv2.perspectiveTransform(pts, homography).astype(np.int)
#     else:
#         dst = cv2.perspectiveTransform(pts, homography).astype(np.int)
#         filter.predict(dst)

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


def render(img: Array[int],
           obj: OBJ,
           scale_factor: float,
           projection: Array[float],
           marker_shape: Tuple[int, int],
           color: Union[bool, Sequence[int]] = False) -> Array[int]:
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale_factor
    h, w = marker_shape
    tmp_image = np.zeros_like(img)

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        # print(points)
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(tmp_image, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(tmp_image, imgpts, color)
    return np.uint8(np.where(tmp_image == 0, 1, 0.5) * img) + tmp_image//2


def mask_from_contours(contours: Sequence[Tuple[int, int]], image: Array[int]) -> Array[int]:
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask.astype(np.uint8)
