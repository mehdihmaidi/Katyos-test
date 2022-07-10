import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np
from model import teachable_machine_classification
import time
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im
from io import StringIO
from tensorflow.keras.preprocessing import image as image_utils
@st.cache(hash_funcs={StringIO: StringIO.getvalue}, suppress_st_warning=True)
def object_detection_image(uploaded_file):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
    holistic=mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete)

    label = teachable_machine_classification(image)

    image2 = np.array(image)
    
        #st.write("Your face shape is ",label)

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        results = holistic.process(image2)
    mp_drawing.draw_landmarks(image2, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
 
    return label,image,image2
def object_detection_video():
    uploaded_file = st.file_uploader('Upload video', type = ['jpg','png','jpeg'])
    
    
    