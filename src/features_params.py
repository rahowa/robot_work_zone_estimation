import os
from src.render import RenderZone
import sys
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from streamlit import sidebar as sb

sys.path.append('./src')
from src.obj_loader import OBJ
from src.calibrate_camera_utils import (CameraParams, save_camera_params,
                                        load_camera_params, calibrate_camera)
from src.aruco_zone_estimation import ArucoZoneEstimator, ARUCO_MARKER_SIZE
from src.common_params import (save_config, get_cap, get_camera_params,
                               get_calibration_params, get_model_params)


def get_marker_params(frame_height: int) -> Tuple[int, str, str]:
    sb.markdown("Params of marker")
    marker_size = sb.number_input("Size of marker",
                                  min_value=6,
                                  max_value=frame_height,
                                  value=64)
    path_to_marker = sb.text_input("Write path to custom marker in .jpg or .png formats",
                                   "./src/data/markers/marker23.png")
    full_path_to_marker = os.path.abspath(path_to_marker)
    path_to_obj = sb.text_input("Write path to custom area form",
                                "./src/data/3dmodels/Cylinder.obj")
    full_path_to_obj = os.path.abspath(path_to_obj)
    return marker_size, full_path_to_marker, full_path_to_obj


def get_detector_params() -> Tuple[float, int, int]:
    sb.markdown("Params of detector")
    scale_factor_feat_det = sb.slider('Chose scale factor for feature detector', 1.01, 1.3, 1.01)
    max_num_of_features = sb.slider("Select number of features", 30, 5000, 1000)
    num_of_levels = sb.number_input("Write number of level for feature detector",
                                    min_value=2, max_value=16, value=16)
    return scale_factor_feat_det, max_num_of_features, num_of_levels


def main_features() -> None:
    st.title("Configure params for worzkone")
    sb.markdown("Params of rendering")
    draw_matches = sb.checkbox("Draw matches", value=False)
    frame_width, frame_height = get_camera_params()
    marker_size, full_path_to_marker, full_path_to_obj = get_marker_params(frame_height)
    scale_factor_feat_det, max_num_of_features, num_of_levels = get_detector_params()
    scale_factor_model = get_model_params()
    board_height, board_width, path_to_images, path_to_camera_params = get_calibration_params()
    
    camera_params: Optional[CameraParams] = None
    if os.path.isfile(path_to_camera_params):
        camera_params = load_camera_params(path_to_camera_params)
    else:
        st.warning("Estimate camera params!")
        st.warning("Press 'Calibrate camera' button")

    if sb.button("Calibrate camera"):
        st.write("Start camera calibration process")
        camera_params = calibrate_camera(path_to_images, 
                                         frame_height,
                                         frame_width,
                                         board_height,
                                         board_width)
        save_camera_params(camera_params, path_to_camera_params)
        # process.join()
        st.write("Camera was calibarted")

    viewer = st.image(np.zeros((frame_height, frame_width, 3)))

    cap = get_cap()
    cap.set(3, frame_height)
    cap.set(4, frame_width)

    params = dict(
        frame_width=frame_width,
        frame_height=frame_height,
        marker_size=marker_size,
        path_to_marker=full_path_to_marker,
        path_to_obj=full_path_to_obj,
        scale_factor_model=scale_factor_model,
        scale_factor_feat_det=scale_factor_feat_det,
        max_num_of_features=max_num_of_features,
        num_of_levels=num_of_levels
    )
    st.json(params)

    if sb.button('Save config'):
        save_config("CustomConfig.json", params)
        st.write('Config saved YOUR_CUSTOM_CONFIG.json')

    if sb.button('Save and Exit'):
        st.write('Exit configure')
        save_config("CustomConfig.json", params)
        cap.release()
        exit(0)

    workzone_model = OBJ("src/data/3dmodels/Cylinder.obj", swapyz=True)
    renderer = RenderZone(workzone_model, scale_factor_model, (7, 7))
    estimator = ArucoZoneEstimator(0.045, ARUCO_MARKER_SIZE['7x7'], camera_params)
    while True:
        ret, scene = cap.read()
        if not ret:
            break
        
        scene = estimator.estimate(scene, None)
        viewer.image(scene, channels='BGR')
    cap.release()

