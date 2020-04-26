import sys
import cv2
import numpy as np
from nptyping import Array
from typing import Optional
from frozendict import frozendict

sys.path.append('./src')
from src.obj_loader import OBJ
from src.counter_filter import CounterFilter
from src.render import RenderZone
from src.utills import compute_corner, draw_corner, projection_matrix
from src.calibrate_camera import CameraParams


ARUCO_MARKER_SIZE = frozendict({
    "4x4": cv2.aruco.DICT_4X4_100,
    "5x5": cv2.aruco.DICT_5X5_100,
    "6x6": cv2.aruco.DICT_6X6_100,
    "7x7": cv2.aruco.DICT_7X7_100,
})


class ArucoZoneEstimator:
    def __init__(self, marker_world_size: float, marker_size: int, camera_params: CameraParams) -> None:
        self.marker_world_size = marker_world_size
        self.marker_size = marker_size
        self.camera_params = camera_params


    def estimate(self, 
                 scene: Array[np.uint8], 
                 renderer: Optional[RenderZone]) -> Array[np.uint8]:
        dictionary = cv2.aruco.Dictionary_get(self.marker_size)
        parameters =  cv2.aruco.DetectorParameters_create()
        counter_filter = CounterFilter(10)
        gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_scene,
                                                                dictionary, 
                                                                parameters=parameters,
                                                                cameraMatrix=self.camera_params.camera_mtx, 
                                                                distCoeff=self.camera_params.distortion_vec)
        if len(marker_corners) > 0:
            counter_filter.init(marker_corners)
        else:
            marker_corners = counter_filter.get()
        
        if len(marker_corners) > 0:
            scene = cv2.aruco.drawDetectedMarkers(scene, marker_corners, marker_ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 
                                                                  self.marker_world_size , 
                                                                  self.camera_params.camera_mtx, 
                                                                  self.camera_params.distortion_vec)
            for idx in range(len(rvecs)):
                scene = cv2.aruco.drawAxis(scene, 
                                           self.camera_params.camera_mtx, 
                                           self.camera_params.distortion_vec,
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
                corner = compute_corner((self.marker_size, self.marker_size), homography)
                scene = draw_corner(scene, corner)
                if renderer is not None:
                    projection = projection_matrix(self.camera_params.camera_mtx, homography)
                    scene = renderer.render(scene, projection)
        return scene