from dataclasses import make_dataclass
import sys
import cv2
import numpy as np
from nptyping import Array
from typing import List, Tuple, Optional
from frozendict import frozendict

sys.path.append('./src')
from src.counter_filter import CounterFilter
from src.utills import compute_corner
from src.calibrate_camera_utils import CameraParams
from src.workzone import Workzone


ARUCO_MARKER_SIZE = frozendict({
    "4x4": cv2.aruco.DICT_4X4_100,
    "5x5": cv2.aruco.DICT_5X5_100,
    "6x6": cv2.aruco.DICT_6X6_100,
    "7x7": cv2.aruco.DICT_7X7_100,
})


class ArucoZoneEstimator:
    def __init__(self,
                 marker_world_size: float,
                 marker_size: int,
                 marker_id: int,
                 camera_params: Optional[CameraParams],
                 zone: Workzone) -> None:
        print("INITED AGAIN")
        self.marker_world_size = marker_world_size
        self.marker_size = marker_size
        self.marker_id = marker_id
        self.camera_params = camera_params
        self.counter_filter = CounterFilter(10)
        self.zone = zone

    def estimate(self, scene: Array[np.uint8]) -> List[Tuple[int, int]]:
        gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        camera_matrix = self.camera_params.camera_mtx if self.camera_params is not None else None
        dist_coeffs =  self.camera_params.distortion_vec if self.camera_params is not None else None
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_scene,
                                                                cv2.aruco.Dictionary_get(self.marker_size),
                                                                parameters=cv2.aruco.DetectorParameters_create(),
                                                                cameraMatrix=camera_matrix,
                                                                distCoeff=dist_coeffs)

        if (marker_ids is None) or (not self.marker_id in marker_ids):
            corner = self.zone.to_polygon()
            return [(int(point.x + self.zone.cx), int(point.y + self.zone.cy))
                    for point in corner]

        if len(marker_corners) > 0:
            self.counter_filter.init(marker_corners)
        else:
            marker_corners = self.counter_filter.get()

        if len(marker_corners) > 0:
            scene = cv2.aruco.drawDetectedMarkers(scene, marker_corners, marker_ids)

            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners,
                                                                      self.marker_world_size,
                                                                      camera_matrix,
                                                                      dist_coeffs)
                for idx in range(len(rvecs)):
                    scene = cv2.aruco.drawAxis(scene,
                                               camera_matrix,
                                               dist_coeffs,
                                               rvecs[idx],
                                               tvecs[idx],
                                               0.03)
            src_points = np.array([[0, 0],
                                   [self.marker_size, 0],
                                   [self.marker_size, self.marker_size],
                                   [0, self.marker_size]])
            src_points = src_points.reshape(-1, 1, 2)
            dst_points = marker_corners[0]
            homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, maxIters=5000)
            if homography is not None:
                corner = compute_corner((self.zone.height, self.zone.width), homography).tolist()
                return [(point[0][0] + self.zone.cx, point[0][1] + self.zone.cy)
                        for point in corner]
            else:
                corner = self.zone.to_polygon()
                return [(int(point.x + self.zone.cx), int(point.y + self.zone.cy))
                        for point in corner]
        else:
            corner = self.zone.to_polygon()
            return [(int(point.x + self.zone.cx), int(point.y + self.zone.cy))
                    for point in corner]
