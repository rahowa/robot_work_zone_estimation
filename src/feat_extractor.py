import cv2
from nptyping import Array
from typing import Tuple, List, Callable, Sequence
from src.utills import mask_from_contours
import numpy as np


class MakeDescriptor:
    def __init__(self, 
                 feature_detector: cv2.Feature2D,
                 marker_path: str,
                 w: int = 512,
                 h: int = 512,
                 img_processing: Callable = None):
        self.detector = feature_detector
        marker = cv2.imread(marker_path)
        wh_ratio = marker.shape[1]/marker.shape[0]
        self.marker = cv2.resize(marker, (int(w * wh_ratio), h))

        if img_processing is not None:
            self.marker = img_processing(self.marker)

        kp_des = self.detector.detectAndCompute(self.marker, None)
        self.kp_marker = kp_des[0]
        self.des_marker = kp_des[1]

    def get_marker_size(self) -> Tuple[int, int]:
        return self.marker.shape[:2]
    
    def get_marker_data(self) -> Tuple[List[cv2.KeyPoint], List[cv2.DMatch]]:
        return self.kp_marker, self.des_marker

    def get_frame_data(self, frame: Array[int],
                       mask: Array[int]) -> Tuple[List[cv2.KeyPoint], List[cv2.DMatch]]:
        kp_frame = self.detector.detect(frame, mask)
        if len(kp_frame) < 1:
            return None, None
        else:
            return self.detector.compute(frame, kp_frame)
