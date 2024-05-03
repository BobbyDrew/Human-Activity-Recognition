# Import necessary libraries
import sys

# Check if the Python version is 3.x or later
if sys.version_info[0] >= 3:
    import io
    
    # Set UTF-8 encoding for stdout and stderr
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
else:
    # For Python 2.x, use reload and setdefaultencoding
    reload(sys)
    sys.setdefaultencoding('utf-8')

# Import necessary libraries
import streamlit as st
from pytube import YouTube
import cv2
import numpy as np
from collections import deque
import os
import base64
from tensorflow.keras.models import load_model

# Load the model
model_file_path = "convlstm_model_89.h5"  # Adjust the path as necessary
try:
    convlstm_model = load_model(model_file_path)
except Exception as e:
    st.error(f"Failed to load the model due to: {e}")
    convlstm_model = None

# Constants for the application
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
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                   int(video_reader.get(cv2.CAP_PROP_FPS)), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH and convlstm_model is not None:
            predicted_labels_probabilities = convlstm_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()
    return output_video_file_path

def get_binary_file_downloader_html(file_path, title="Download File"):
    try:
        with open(file_path, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode()
        return f'<a href="data:file/mp4;base64,{b64}" download="{os.path.basename(file_path)}">{title}</a>'
    except FileNotFoundError:
        return "File not found. Please ensure the file was created successfully."

def main():
    st.title("Human Activity Prediction")
    if st.sidebar.checkbox("Overview of the Project"):
        st.write("This project aims to predict human activities such as Biking, Diving, Golf Swing, and Pizza Tossing from videos using a convolutional LSTM model.")

    upload_type = st.sidebar.selectbox("How would you like to upload a video?", ("YouTube URL", "Local Device", "Live Camera"))
    if upload_type == "YouTube URL":
        youtube_url = st.text_input("Enter the YouTube URL:")
        if youtube_url and st.button("Download YouTube Video"):
            test_videos_directory = 'test_videos'
            os.makedirs(test_videos_directory, exist_ok=True)
            video_file_path, video_title = download_youtube_video(youtube_url, test_videos_directory)
            output_video_file_path = os.path.join(test_videos_directory, f"{video_title}-output.mp4")
            output_video_file_path = perform_action_recognition(video_file_path, output_video_file_path)
            st.success("Prediction complete! You can download the output video below.")
            st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)

    elif upload_type == "Local Device":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4'])
        if uploaded_file is not None:
            test_videos_directory = 'test_videos'
            os.makedirs(test_videos_directory, exist_ok=True)
            video_file_path = os.path.join(test_videos_directory, uploaded_file.name)
            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.read())
            output_video_file_path = os.path.join(test_videos_directory, f"{uploaded_file.name}-output.mp4")
            output_video_file_path = perform_action_recognition(video_file_path, output_video_file_path)
            st.success("Prediction complete! You can download the output video below.")
            st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)

    elif upload_type == "Live Camera":
        st.write("Live camera feature is not implemented in this script.")

if __name__ == "__main__":
    main()
