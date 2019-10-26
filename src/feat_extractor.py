import math
import cv2 
import numpy as np
from typing import Tuple, List, Sequence, Union, Any

class MakeDescriptor:
    def __init__(self, 
                feature_detector: cv2.Feature2D, 
                marker_path: str,
                w:int = 512,
                h:int = 512):
        self.detector = feature_detector
        marker = cv2.imread(marker_path)
        self.marker = cv2.resize(marker, (w, h))
        kp_marker = self.detector.detect(self.marker)
        kp_des = self.detector.compute(self.marker, kp_marker)

        self.kp_marker = kp_des[0]
        self.des_marker = kp_des[1]

    def get_marker_size(self) -> List[int]:
        return self.marker.shape[:2]
    
    def get_marker_data(self) -> Tuple[List[cv2.KeyPoint], List[cv2.DMatch]]:
        return self.kp_marker, self.des_marker

    def get_frame_data(self, frame) -> Tuple[List[cv2.KeyPoint], List[cv2.DMatch]]:
        kp_frame = self.detector.detect(frame)
        return self.detector.compute(frame, kp_frame)