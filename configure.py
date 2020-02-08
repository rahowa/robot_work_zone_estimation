import os
import sys
import json
from typing import Dict, Any

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


params = dict()
def save_config(params: Dict[str, Any]):
    with open('YOUR_CUSTOM_CONFIG.json', 'w') as json_file:
        json.dump(params, json_file)

@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)

# '''Params of rendering and perfomance'''
st.title("Configure params for worzkone")
sb.markdown("Params of rendering")
frameST = st.empty() 

draw_matches = sb.checkbox("Chose to draw matches or not",
                           value=False)
frame_width = sb.number_input("Width of input frame",
                              min_value=120,
                              max_value=1920,
                              value=640)
frame_height = int(3/4 * frame_width)
focal_lenght = sb.number_input("Focal lenght of camera",
                               min_value=0.0,
                               max_value=2000.0,
                               value=800.0)
marker_size = sb.number_input("Size of marker",
                              min_value=16, max_value=frame_height,
                              value=200)
path_to_marker = sb.text_input("Write path to custom marker in .jpg or .png formats",
                               "./src/data/markers/test_column.jpg")
full_path_to_marker = os.path.abspath(path_to_marker)
path_to_obj = sb.text_input("Write path to custom area form",
                            "./src/data/3dmodels/Cylinder.obj")
full_path_to_obj = os.path.abspath(path_to_obj)
scale_factor_model = sb.number_input("Write scaled factor for 3d model of workzone",
                                     1e-4, 1e4, (100.0))

# '''Params of feature detector and descriptor'''
sb.markdown("Params of detector")
scale_factor_feat_det = sb.slider('Chose scale factor for feature detector',
                                  1.0, 3.0, (1.1))

max_num_of_features = sb.slider("Select number of features",
                                30, 5000, (1000)
)

num_of_levels = sb.number_input("Write number of level for feature detector",
                                min_value=2, max_value=100,
                                value=16)

cap = get_cap()
cap.set(3, frame_height)
cap.set(4, frame_width)

params['frame_width'] = frame_width
params['frame_height'] = frame_height
params['focal_lenght'] = focal_lenght
params['marker_size'] = marker_size
params['path_to_marker'] = full_path_to_marker
params['path_to_obj'] = full_path_to_obj
params['scale_factor_model'] = scale_factor_model
params['scale_factor_feat_det'] = scale_factor_feat_det
params['max_num_of_features'] = max_num_of_features
params['num_of_levels'] = num_of_levels

if sb.button('Save config'):
    save_config(params)
    st.write('Config saved YOUR_CUSTOM_CONFIG.json')

if sb.button('Save and Exit'):
   st.write('Exit configure')
   save_config(params)
   exit(0)

CAMERA_PARAMS = np.array([[focal_lenght , 0,            frame_width//2], 
                          [0,             focal_lenght, frame_height//2], 
                          [0,             0,            1]])

obj = OBJ(full_path_to_obj, swapyz=True)
descriptor_params   = dict(nfeatures=max_num_of_features, scaleFactor=scale_factor_feat_det, 
                         nlevels=num_of_levels, edgeThreshold=10, firstLevel=0)
homography_alg      = ComputeHomography(cv2.BFMatcher_create(cv2.NORM_HAMMING, 
                                   crossCheck=True))
column_descriptor   = MakeDescriptor(cv2.ORB_create(**descriptor_params), 
                                   full_path_to_marker, marker_size, marker_size)
while True:
    _, scene  = cap.read()
    kp_marker, des_marker   = column_descriptor.get_marker_data()
    kp_scene, des_scene     = column_descriptor.get_frame_data(scene)
    if des_marker is not None and des_scene is not None:
        homography = homography_alg(kp_scene, kp_marker, des_scene, des_marker)
        
        if homography is not None:
            scene       = draw_corner(scene, column_descriptor.get_marker_size(), homography)
            projection  = projection_matrix(CAMERA_PARAMS, homography)
            scene       = render(scene, obj, scale_factor_model, 
                            projection, column_descriptor.get_marker_size(), False)
    
        if homography_alg.matches is not None and draw_matches:
            scene = cv2.drawMatches(column_descriptor.marker,
                                    kp_marker,
                                    scene, 
                                    kp_scene, 
                                    homography_alg.matches, 
                                    0, flags=2)
    frameST.image(scene, channels='BGR')

cap.release()
