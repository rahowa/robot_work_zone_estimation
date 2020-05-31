from argparse import ArgumentError
import cv2
from nptyping import Array
from typing import Tuple, List, Callable, Union
import numpy as np


class MakeDescriptor:
    def __init__(self, 
                 feature_detector: cv2.Feature2D,
                 marker: Union[str, Array[np.uint8]],
                 w: int = 512,
                 h: int = 512,
                 img_processing: Callable = None):
        self.detector = feature_detector

        if isinstance(marker, str):
            loaded_marker = cv2.imread(marker)
            wh_ratio = loaded_marker.shape[1]/loaded_marker.shape[0]
            self.marker = cv2.resize(loaded_marker, (int(w * wh_ratio), h))
        elif isinstance(marker, Array[np.uint8]):
            self.marker = marker
        else:
            raise ArgumentError 


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
