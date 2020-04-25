import os
import sys
import json
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
from nptyping import Array
import streamlit as st
from streamlit import sidebar as sb
import multiprocessing as mp

sys.path.append('./src')
from src.obj_loader import OBJ
from src.utills import (CounterFilter)
from src.calibrate_camera import CameraParams, save_camera_params, load_camera_params, calibrate_camera


def save_config(config_name: str, params: Dict[str, Any]) -> None:
    with open(config_name, 'w') as json_file:
        json.dump(params, json_file)


@st.cache(allow_output_mutation=True)
def get_cap() -> cv2.VideoCapture:
    capture =  cv2.VideoCapture(0)
    return capture


def get_camera_params() -> Tuple[int, int, int]:
    sb.markdown("Params of camera")
    frame_width = sb.number_input("Width of input frame",
                                  min_value=120,
                                  max_value=1920,
                                  value=640)
    frame_height = int(3 / 4 * frame_width)
    focal_length = sb.number_input("Focal lenght of camera",
                                   min_value=0.0,
                                   max_value=2000.0,
                                   value=800.0)
    return frame_width, frame_height, focal_length


def get_calibration_params() -> Tuple[int, int, str, str]:
    sb.markdown("Params of cameara calibration porcesss")

    path_to_camera_params = sb.text_input("Path to camera params",
                                          "camera_params.json")
    board_height = sb.number_input("Board height", 
                                    min_value=3,
                                    max_value=100,
                                    value=9)
    board_width = sb.number_input("Board width", 
                                    min_value=3,
                                    max_value=100,
                                    value=6)
    path_to_images = sb.text_input("Path to images with board",
                                    "./src/data/calibration")
    return board_height, board_width, path_to_images, path_to_camera_params


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


def get_model_params() -> float:
    sb.markdown("Params of 3d model")
    return sb.number_input("Write scaled factor for 3d model of workzone",
                           1e-4, 1e4, 10.0)


def get_detector_params() -> Tuple[float, int, int]:
    sb.markdown("Params of detector")
    scale_factor_feat_det = sb.slider('Chose scale factor for feature detector', 1.01, 1.3, 1.01)
    max_num_of_features = sb.slider("Select number of features", 30, 5000, 1000)
    num_of_levels = sb.number_input("Write number of level for feature detector",
                                    min_value=2, max_value=16, value=16)
    return scale_factor_feat_det, max_num_of_features, num_of_levels


def preprocess_image(img: Array[np.uint8]) -> Array[np.uint8]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getGaussianKernel(5, 0)
    img = cv2.filter2D(img, cv2.CV_8U, kernel)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 0.0)
    return img


def main() -> None:
    st.title("Configure params for worzkone")
    sb.markdown("Params of rendering")
    draw_matches = sb.checkbox("Draw matches", value=False)
    frame_width, frame_height, focal_length = get_camera_params()
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
        focal_lenght=focal_length,
        marker_size=marker_size,
        path_to_marker=full_path_to_marker,
        path_to_obj=full_path_to_obj,
        scale_factor_model=scale_factor_model,
        scale_factor_feat_det=scale_factor_feat_det,
        max_num_of_features=max_num_of_features,
        num_of_levels=num_of_levels
    )

    if sb.button('Save config'):
        save_config("CustomConfig.json", params)
        st.write('Config saved YOUR_CUSTOM_CONFIG.json')

    if sb.button('Save and Exit'):
        st.write('Exit configure')
        save_config("CustomConfig.json", params)
        cap.release()
        exit(0)

    prev_corners = None
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_1000)
    parameters =  cv2.aruco.DetectorParameters_create()
    prev_corners = None
    counter_filter = CounterFilter(10)
    while True:
        ret, scene = cap.read()
        if not ret:
            break

        gray_scene = cv2.cvtColor(scene.copy(), cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_scene,
                                                                dictionary, 
                                                                parameters=parameters,
                                                                cameraMatrix=camera_params.camera_mtx, 
                                                                distCoeff=camera_params.distortion_vec)
        if len(marker_corners) > 0:
            counter_filter.init(marker_corners)
        elif len(marker_corners) == 0 and prev_corners is not None:
            marker_corners = counter_filter.get()
        
        # prev_corners = marker_corners
        if len(marker_corners) > 0:
            scene = cv2.aruco.drawDetectedMarkers(scene.copy(), marker_corners, marker_ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 0.045 , 
                                                                  camera_params.camera_mtx, 
                                                                  camera_params.distortion_vec)
            for idx in range(len(rvecs)):
                scene = cv2.aruco.drawAxis(scene, 
                                           camera_params.camera_mtx, 
                                           camera_params.distortion_vec,
                                           rvecs[idx], 
                                           tvecs[idx], 
                                           0.03)

        
        viewer.image(scene, channels='BGR')
    cap.release()


if __name__ == "__main__":
    main()
