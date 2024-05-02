import streamlit as st
from pytube import YouTube
import cv2
import numpy as np
from collections import deque
import os
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal  # If you need custom initializers

# Load the model
model_file_path = "convlstm_model_89.h5"  # Ensure this path is accessible in your deployment environment
convlstm_model = load_model(model_file_path)

# Define constants
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

def perform_action_recognition(video_file_path, output_file_path):
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   int(video_reader.get(cv2.CAP_PROP_FPS)), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = convlstm_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
            annotate_frame(frame, predicted_class_name)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()
    return output_file_path

def annotate_frame(frame, text):
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x, text_y = 10, 30
    padding = 5
    box_coords = ((text_x, text_y + padding), (text_x + text_size[0] + padding * 2, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)
    cv2.putText(frame, text, (text_x + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def get_binary_file_downloader_html(file_path, title="Download File"):
    with open(file_path, "rb") as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    return f'<a href="data:file/mp4;base64,{b64}" download="{os.path.basename(file_path)}">{title}</a>'

def main():
    st.title("Human Activity Prediction")
    overview = st.sidebar.checkbox("Overview of the Project")
    if overview:
        show_overview()

    add_selectbox = st.sidebar.selectbox("How would you like to upload a video?", ("YouTube URL", "Local Device", "Live Camera"))
    handle_video_upload(add_selectbox)

def show_overview():
    st.header("Project Overview")
    st.write("""
        This project aims to predict human activity from videos using a convolutional LSTM model. 
        Activities such as Biking, Diving, Golf Swing, and Pizza Tossing can be recognized.
    """)
    st.image('path_to_images_directory', caption='Activity Examples', use_column_width=True)

def handle_video_upload(upload_type):
    if upload_type == "YouTube URL":
        youtube_url = st.text_input("Enter the YouTube URL:")
        if youtube_url and st.button("Download YouTube Video"):
            process_youtube_video(youtube_url)
    elif upload_type == "Local Device":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4'])
        if uploaded_file:
            process_local_video(uploaded_file)
    elif upload_type == "Live Camera":
        st.write("Live camera feed not implemented in this script.")

def process_youtube_video(youtube_url):
    test_videos_directory = 'test_videos'
    os.makedirs(test_videos_directory, exist_ok=True)
    video_file_path, video_title = download_youtube_video(youtube_url, test_videos_directory)
    st.success(f"Download complete! Video saved as: {video_title}.mp4")
    process_video(test_videos_directory, video_file_path, video_title)

def process_local_video(uploaded_file):
    test_videos_directory = 'test_videos'
    os.makedirs(test_videos_directory, exist_ok=True)
    video_file_path = os.path.join(test_videos_directory, uploaded_file.name)
    with open(video_file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"You uploaded: {uploaded_file.name}")
    process_video(test_videos_directory, video_file_path, uploaded_file.name)

def process_video(directory, video_file_path, title):
    output_video_file_path = f'{directory}/{title}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
    output_video_file_path = perform_action_recognition(video_file_path, output_video_file_path)
    st.success("Prediction complete! You can download the output video below.")
    st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
