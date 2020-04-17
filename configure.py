import os
import sys
import json
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import streamlit as st
from streamlit import sidebar as sb

sys.path.append('./src')
from src.obj_loader import OBJ
from src.homography import ComputeHomography
from src.feat_extractor import MakeDescriptor
from src.utills import (projection_matrix, render,
                        draw_corner)


def save_config(config_name: str, params: Dict[str, Any]) -> None:
    with open(config_name, 'w') as json_file:
        json.dump(params, json_file)


@st.cache(allow_output_mutation=True)
def get_cap() -> cv2.VideoCapture:
    return cv2.VideoCapture(0)


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


def get_marker_params(frame_height: int) -> Tuple[int, str, str]:
    sb.markdown("Params of marker")
    marker_size = sb.number_input("Size of marker",
                                  min_value=16,
                                  max_value=frame_height,
                                  value=200)
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
                           1e-4, 1e4, 100.0)


def get_detector_params() -> Tuple[float, int, int]:
    sb.markdown("Params of detector")
    scale_factor_feat_det = sb.slider('Chose scale factor for feature detector', 1.0, 3.0, 1.1)
    max_num_of_features = sb.slider("Select number of features", 30, 5000, 1000)
    num_of_levels = sb.number_input("Write number of level for feature detector",
                                    min_value=2, max_value=100, value=16)
    return scale_factor_feat_det, max_num_of_features, num_of_levels


def main() -> None:
    st.title("Configure params for worzkone")
    sb.markdown("Params of rendering")
    draw_matches = sb.checkbox("Draw matches", value=False)
    frame_width, frame_height, focal_length = get_camera_params()
    marker_size, full_path_to_marker, full_path_to_obj = get_marker_params(frame_height)
    scale_factor_feat_det, max_num_of_features, num_of_levels = get_detector_params()
    scale_factor_model = get_model_params()

    viewer = st.image(np.zeros((frame_height, frame_width, 3)))
    camera_params = np.array([[focal_length, 0, frame_width // 2],
                              [0, focal_length, frame_height // 2],
                              [0, 0, 1]])

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

    obj = OBJ(full_path_to_obj, swapyz=True)
    descriptor_params = dict(nfeatures=max_num_of_features,
                             scaleFactor=scale_factor_feat_det,
                             nlevels=num_of_levels,
                             edgeThreshold=10,
                             firstLevel=0)
    homography_alg = ComputeHomography(cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True))
    column_descriptor = MakeDescriptor(cv2.ORB_create(**descriptor_params),
                                       full_path_to_marker,
                                       marker_size,
                                       marker_size)
    while True:
        ret, scene = cap.read()
        if not ret:
            continue
            
        kp_marker, des_marker = column_descriptor.get_marker_data()
        kp_scene, des_scene = column_descriptor.get_frame_data(scene)
        if des_marker is not None and des_scene is not None:
            homography = homography_alg(kp_scene, kp_marker, des_scene, des_marker)

            if homography is not None:
                scene = draw_corner(scene, column_descriptor.get_marker_size(), homography)
                projection = projection_matrix(camera_params, homography)
                scene = render(scene, obj, scale_factor_model,
                               projection, column_descriptor.get_marker_size(), False)

            if homography_alg.matches is not None and draw_matches:
                scene = cv2.drawMatches(column_descriptor.marker,
                                        kp_marker,
                                        scene,
                                        kp_scene,
                                        homography_alg.matches,
                                        0, flags=2)
        viewer.image(scene, channels='BGR')
    cap.release()


if __name__ == "__main__":
    main()
