import os
from src.render import RenderZone
import sys
import json
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from streamlit import sidebar as sb

sys.path.append('./src')
from src.obj_loader import OBJ
from src.calibrate_camera import (CameraParams, save_camera_params,
                                  load_camera_params, calibrate_camera)
from src.aruco_zone_estimation import ArucoZoneEstimator, ARUCO_MARKER_SIZE
from src.common_params import (save_config, get_cap,get_camera_params,
                               get_calibration_params, get_model_params)


def get_aruco_marker_params() -> Tuple[str, float]:
    sb.markdown("Params of marker")
    marker_size = sb.selectbox("Chose aruco marker size",
                                ("4x4", "5x5", "6x6", "7X7"))
    marker_world_size = sb.number_input("Lenght of real world marker",
                                        min_value=0.0,
                                        max_value=100.0,
                                        value=0.05)
    return marker_size, marker_world_size


def get_detector_params() -> Tuple[float, int, int]:
    sb.markdown("Params of detector")
    scale_factor_feat_det = sb.slider('Chose scale factor for feature detector', 1.01, 1.3, 1.01)
    max_num_of_features = sb.slider("Select number of features", 30, 5000, 1000)
    num_of_levels = sb.number_input("Write number of level for feature detector",
                                    min_value=2, max_value=16, value=16)
    return scale_factor_feat_det, max_num_of_features, num_of_levels


def main_aruco() -> None:
    st.title("Configure params for worzkone")
    sb.markdown("Params of rendering")
    frame_width, frame_height = get_camera_params()
    marker_size, marker_world_size = get_aruco_marker_params()
    scale_factor_model = get_model_params()
    camera_params: Optional[CameraParams] = None
    show_calibration_params = sb.checkbox("Camera calibration")
    if show_calibration_params:
        board_height, board_width, path_to_images, path_to_camera_params = get_calibration_params()
        if os.path.isfile(path_to_camera_params):
            camera_params = load_camera_params(path_to_camera_params)
        else:
            st.warning("Estimate camera params!")
            st.warning("Press 'Calibrate camera' button")

        if sb.button("Calibrate cemera"):
            st.write("Start camera calibration process")
            with st.spinner("Calibrating camera..."):
                camera_params = calibrate_camera(path_to_images, 
                                                 frame_height,
                                                 frame_width,
                                                 board_height,
                                                 board_width)
            save_camera_params(camera_params, path_to_camera_params)
            st.success("Camera was calibrated successfully")
            st.write("Camera was calibarted")

    viewer = st.image(np.zeros((frame_height, frame_width, 3)))

    cap = get_cap()
    cap.set(3, frame_height)
    cap.set(4, frame_width)

    params = dict(
        frame_width=frame_width,
        frame_height=frame_height,
        marker_size=marker_size,
        marker_world_size=marker_world_size,
        scale_factor_model=scale_factor_model,
        camera_params=camera_params.to_dict() if camera_params is not None else {}
    )

    st.subheader("Zone estimation configuration:")
    st.json(params)

    path_to_config = sb.text_input("Path to config", "aruco_config.json")
    if sb.button('Save config'):
        save_config(path_to_config, params)
        st.write(f'Config saved to {path_to_config}')

    if sb.button('Save and Exit'):
        save_config(path_to_config, params)
        st.write('Exit configure')
        cap.release()
        exit(0)

    workzone_model = OBJ("src/data/3dmodels/Cylinder.obj", swapyz=True)
    renderer = RenderZone(workzone_model, scale_factor_model, (7, 7))
    estimator = ArucoZoneEstimator(marker_world_size, 
                                   ARUCO_MARKER_SIZE[marker_size],
                                   camera_params)
    while True:
        ret, scene = cap.read()
        if not ret:
            break
        
        scene = estimator.estimate(scene, None)
        viewer.image(scene, channels='BGR')
    cap.release()
