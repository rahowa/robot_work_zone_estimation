import sys
import json
from typing import Dict, Any, Tuple

import cv2
import streamlit as st
from streamlit import sidebar as sb

sys.path.append('./src')
from src.obj_loader import OBJ


def save_config(config_name: str, params: Dict[str, Any]) -> None:
    with open(config_name, 'w') as json_file:
        json.dump(params, json_file)


@st.cache(allow_output_mutation=True)
def get_cap() -> cv2.VideoCapture:
    capture =  cv2.VideoCapture(0)
    return capture


def get_camera_params() -> Tuple[int, int]:
    sb.markdown("Params of camera")
    frame_width = sb.number_input("Width of input frame",
                                  min_value=120,
                                  max_value=1920,
                                  value=640)
    frame_height = int(3 / 4 * frame_width)
    return frame_width, frame_height


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


def get_model_params() -> float:
    sb.markdown("Params of 3d model")
    return sb.number_input("Write scaled factor for 3d model of workzone",
                           1e-4, 1e4, 10.0)
