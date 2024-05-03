import streamlit as st
from pytube import YouTube
import cv2
import numpy as np
from collections import deque
import os
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal

# Define custom objects if your model uses any
custom_objects = {'Orthogonal': Orthogonal(gain=1.0, seed=None)}

# Attempt to load the model
model_file_path = "convlstm_model_89.h5"
try:
    convlstm_model = load_model(model_file_path, custom_objects=custom_objects)
except Exception as e:
    print(f"Error loading the model: {e}")
    convlstm_model = None  # Handle cases where the model isn't loaded

# Constants
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["Biking", "Diving", "Golf Swing", "Pizza Tossing"]

def download_youtube_video(youtube_url, output_directory):
    yt = YouTube(youtube_url)
    stream = yt.streams.get_highest_resolution()
    output_file_path = os.path.join(output_directory, f"{yt.title}.mp4")
    stream.download(output_directory)
    return output_file_path, yt.title

def perform_action_recognition(video_file_path, output_video_file_path):
    video_reader = cv2.VideoCapture(video_file_path)
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            if convlstm_model is not None:
                predicted_labels_probabilities = convlstm_model.predict(np.expand_dims(frames_queue, axis=0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = CLASSES_LIST[predicted_label]
                annotate_frame(frame, predicted_class_name)

    video_reader.release()
    return output_video_file_path

def annotate_frame(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def get_binary_file_downloader_html(file_path, title="Download File"):
    with open(file_path, "rb") as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    return f'<a href="data:file/mp4;base64,{b64}" download="{os.path.basename(file_path)}">{title}</a>'

def main():
    st.title("Human Activity Prediction")
    if st.sidebar.checkbox("Overview of the Project"):
        show_overview()

    upload_type = st.sidebar.selectbox("How would you like to upload a video?", ("YouTube URL", "Local Device", "Live Camera"))
    if upload_type == "YouTube URL":
        handle_youtube_upload()
    elif upload_type == "Local Device":
        handle_local_upload()
    elif upload_type == "Live Camera":
        st.write("Live camera feed not implemented in this script.")

def show_overview():
    st.header("Project Overview")
    st.write("This project aims to predict human activity from videos using a convolutional LSTM model.")

def handle_youtube_upload():
    youtube_url = st.text_input("Enter the YouTube URL:")
    if youtube_url and st.button("Download YouTube Video"):
        test_videos_directory = 'test_videos'
        os.makedirs(test_videos_directory, exist_ok=True)
        video_file_path, video_title = download_youtube_video(youtube_url, test_videos_directory)
        st.success(f"Download complete! Video saved as: {video_title}.mp4")
        output_video_file_path = perform_action_recognition(video_file_path, video_file_path.replace('.mp4', '-Output.mp4'))
        st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)

def handle_local_upload():
    uploaded_file = st.file_uploader("Upload a video", type=['mp4'])
    if uploaded_file:
        test_videos_directory = 'test_videos'
        os.makedirs(test_videos_directory, exist_ok=True)
        video_file_path = os.path.join(test_videos_directory, uploaded_file.name)
        with open(video_file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"You uploaded: {uploaded_file.name}")
        output_video_file_path = perform_action_recognition(video_file_path, video_file_path.replace('.mp4', '-Output.mp4'))
        st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
