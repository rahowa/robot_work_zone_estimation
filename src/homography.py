import cv2
import numpy as np
from nptyping import Array
from typing import List


class ComputeHomography:
    def __init__(self, matcher: cv2.DescriptorMatcher, neighbours: int = 3):
        self.matcher = matcher
        self.matches: List[cv2.DMatch] = list()
        self.neighbours = neighbours

    def _find_homography(self,
                         kp_frame: List[cv2.KeyPoint],
                         kp_marker: List[cv2.KeyPoint],
                         matches: List[cv2.DMatch],
                         threshold: float = 5.001,
                         maxiters: int = 10000000) -> Array[float]:
        src_pts = np.float32([kp_marker[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                  ransacReprojThreshold=threshold, maxIters=maxiters)
        return M

    def __call__(self, 
                 kp_frame: List[cv2.KeyPoint],
                 kp_marker: List[cv2.KeyPoint],
                 des_frame: List[cv2.DMatch],
                 des_marker: List[cv2.DMatch],
                 n_best: int = 100,
                 threshold: float = 5.0) -> Array[float]:
        
        if isinstance(self.matcher, cv2.FlannBasedMatcher):
            self.matches = self.matcher.knnMatch(des_marker, des_frame, k=self.neighbours)
        else:
            self.matches = self.matcher.match(des_marker, des_frame)
        self.matches = sorted(self.matches, key=lambda x: x.distance)[:n_best]
        return self._find_homography(kp_frame, kp_marker, self.matches, threshold)
