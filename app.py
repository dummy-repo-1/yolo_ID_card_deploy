
# Importing libraries
import streamlit as st
from ultralytics import YOLO

from PIL import Image, ImageChops, ImageEnhance
import os
import urllib.request
import cv2
import numpy as np
import pandas as pd
import time
import itertools
import glob

MODEL_URL = "https://github.com/dummy-repo-1/yolo_ID_card_deploy/releases/download/weights.v1.0/weights.onnx"
MODEL_PATH = "weights.onnx"
st.set_page_config(page_title="Image Input", layout="centered")
st.title("AI-Powered ID Card Forgery Detection with Precision", text_alignment ="center")
st.html(
    "<div style=font-size:20px;>Uncover hidden tampering in seconds using a powerful fusion of YOLO — delivering 99.5% accurate fraud detection.<br/>You can check out the source code <a href='https://github.com/projjal2025/yolov8-ID-card', target='_blank'>Click here</a></div>"
)
st.subheader("Upload ID card images")
# Taking an image
uploaded_files = st.file_uploader(
    "",
    accept_multiple_files=True,
    type=["jpg", "png"],
    label_visibility="collapsed"
)

# Loading Inference model with trained weights
@st.cache_resource
def load_model():
    # st.info("Downloading model weights for the first time. This may take a minute...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    # st.success("Model downloaded successfully!")
        
    return YOLO(MODEL_PATH, task='detect')


# Function for converting colored and tampered images into ELA images
def convert_to_ela_image(uploaded_files, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    ela_files = []

    for path in uploaded_files :
        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality = quality)
        temp_image = Image.open(temp_filename)

        ela_image = ImageChops.difference(image, temp_image)

        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        ela_files.append(ela_image)

    return ela_files

# function for drawing bounding boxes on colored and tampered images depending
# upon the bounding boxes detected by YOLOv8
def create_refined_bounding_boxes(image, detections):
    refined_boxes = []
    # checking if there are any detections
    if not detections:
        print("No detections provided. Returning original image and empty list.")
        return image, refined_boxes

    # iterating through the detections and draw individual bounding boxes
    for x_min, y_min, x_max, y_max in detections:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 8)  # Red rectangle, thickness 2
        refined_boxes.append((x_min, y_min, x_max, y_max))

    return image, refined_boxes

# function for detecting copy paste boundary artifacts on ELA image by using YOLOv8
# after detecting it will draw bounding boxes on colored and tampered image
def get_predictions(uploaded_files) :

    real_img = "real.jpg"
    count = 0
    forged = 0

    # Getting all ELA converted images
    ela_files = convert_to_ela_image(uploaded_files, 90)

    predicted_files = []

    for path, ela_img in zip(uploaded_files, ela_files):
        # Perform prediction with YOLOv8
        results = model.predict(source=ela_img, show=False, save=False, conf=0.5)

        # checking for bounding boxes -
        for result in results:
            count += 1
            if len(result.boxes) > 0:
                forged += 1
                break  # Exit the loop after the first detection

        
        image = Image.open(path).convert('RGB')
        image.save(real_img)
        cv_real_image = cv2.imread(real_img)

        # extracting bounding box coordinates from YOLOv8 results
        detections = []
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Convert to integers
                detections.append((x_min, y_min, x_max, y_max))

        # creating the refined bounding boxes
        image_with_boxes, refined_boxes = create_refined_bounding_boxes(cv_real_image, detections)  # Pass a copy of the image

        # Coverting BGR image to RGB image
        predicted_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        predicted_files.append(predicted_image)

    return predicted_files, count, forged

st.markdown(
    """
    <style>
        [data-testid="stFileUploaderFileData"] {
            diaplay: none !important;
        }
        [data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderIcon"], [data-testid="stFileUploaderDeleteBtn"], [data-testid="stFileUploaderFileSize"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

model = load_model()

predicted_files, count, forged = get_predictions(uploaded_files)


left_col, right_col = st.columns(2)
if count > 0 : 
    st.subheader(f"Out of {count}, {forged} fake images found.")
    left_col.subheader("Original images", text_alignment="center")
    right_col.subheader("Predicted images", text_alignment="center")
    for real, predicted in zip(uploaded_files, predicted_files) :
        with left_col :
            st.image(real)
        
        with right_col :
            st.image(predicted)

