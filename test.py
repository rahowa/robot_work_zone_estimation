import math
import cv2 
import numpy as np
from typing import Tuple, List, Sequence, Union, Any

from src.obj_loader import OBJ
from src.feat_extractor import MakeDescriptor
from src.homography import ComputeHomography
from src.utills import (projection_matrix, render, 
                    draw_corner)

MIN_MATCHES = 15
CAMERA_PARAMS = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    obj = OBJ("Deer.obj", swapyz=True)
    homography_alg = ComputeHomography(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))
    column_descriptor = MakeDescriptor(cv2.AKAZE_create(), "./external/test_marker.jpg", 640, 480)

    while True:
        _, scene  = cap.read()
        kp_marker, des_marker = column_descriptor.get_marker_data()
        kp_scene, des_scene = column_descriptor.get_frame_data(scene)
        if des_marker is not None:
            homography = homography_alg(kp_scene, kp_marker, des_scene, des_marker)

            if homography is not None:       
                scene = draw_corner(scene, column_descriptor.get_marker_size(), homography)
                projection = projection_matrix(CAMERA_PARAMS, homography)
                scene = render(scene, obj, projection, column_descriptor.get_marker_size(), False)
        cv2.imshow("kp", scene)
        if cv2.waitKey(25) % 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()