import os
from src.render import RenderZone
import sys
import json
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
from nptyping import Array
import streamlit as st
from streamlit import sidebar as sb

sys.path.append('./src')
from src.obj_loader import OBJ
from src.calibrate_camera import (CameraParams, save_camera_params,
                                  load_camera_params, calibrate_camera)
from src.aruco_zone_estimation import ArucoZoneEstimator, ARUCO_MARKER_SIZE
from src.aruco_params import main_aruco


def main() -> None:
    main_aruco()

if __name__ == "__main__":
    main()
