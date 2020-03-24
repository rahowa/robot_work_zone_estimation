import math
import cv2 
import numpy as np
from nptyping import Array
from typing import List, Any

class ComputeHomography:
    def __init__(self, matcher: cv2.DescriptorMatcher):
        self.matcher = matcher
        self.matches: List[cv2.DMatch] = list()

    def _find_homography(self,
                        kp_frame: List[cv2.KeyPoint], 
                        kp_marker: List[cv2.KeyPoint], 
                        matches: List[cv2.DMatch]) -> Array[float]:
        src_pts = np.float32([kp_marker[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M

    def __call__(self, 
                kp_frame: List[cv2.KeyPoint],   
                kp_marker: List[cv2.KeyPoint],
                des_frame: List[cv2.DMatch],
                des_marker: List[cv2.DMatch]) -> Array[float]:
        
        if isinstance(self.matcher, (cv2.FlannBasedMatcher)):
            self.matches = self.matcher.knnMatch(des_marker, des_frame, k=2)
        else:
            self.matches = self.matcher.match(des_marker, des_frame)
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        homography = self._find_homography(kp_frame, kp_marker, self.matches)
        return homography