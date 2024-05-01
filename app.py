import streamlit as st
from pytube import YouTube
import cv2
import numpy as np
from collections import deque
import os
import base64
from moviepy.editor import VideoFileClip
from keras.models import load_model

# Load the model
model_file_path = "C:/Users/veeri/OneDrive - Lovely Professional University/Desktop/Final Project/convlstm_model_89.h5"  # Change this path accordingly
convlstm_model = load_model(model_file_path)

# Define constants
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["Biking","Diving","GolfSwing","PizzaTossing"]

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
            # Perform action recognition
            predicted_labels_probabilities = convlstm_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Draw predicted class name on frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Draw predicted class name on frame with black background box
            text_size = cv2.getTextSize(predicted_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x, text_y = 10, 30  # Position of the text
            padding = 5  # Padding around the text
            box_coords = ((text_x, text_y + padding), (text_x + text_size[0] + padding * 2, text_y - text_size[1] - padding))

            # Draw the black background box
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)

            # Draw the predicted class name on the frame
            cv2.putText(frame, predicted_class_name, (text_x + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

    return output_file_path

def get_binary_file_downloader_html(file_path, title="Download File"):
    with open(file_path, "rb") as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    file_href = f'<a href="data:file/mp4;base64,{b64}" download="{os.path.basename(file_path)}">{title}</a>'
    return file_href

def perform_action_recognition_from_camera(frame, frames_queue):
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            # Perform action recognition (replace this with your actual model prediction)
            # Dummy prediction for demonstration
            predicted_labels_probabilities = convlstm_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.random.randint(len(CLASSES_LIST))
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Draw predicted class name on frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw predicted class name on frame with black background box
            text_size = cv2.getTextSize(predicted_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x, text_y = 10, 30  # Position of the text
            padding = 5  # Padding around the text
            box_coords = ((text_x, text_y + padding), (text_x + text_size[0] + padding * 2, text_y - text_size[1] - padding))

            # Draw the black background box
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), -1)

            # Draw the predicted class name on the frame
            cv2.putText(frame, predicted_class_name, (text_x + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
        
def main():
    st.title("Human Activity Prediction")
    agree= st.sidebar.checkbox("Overview of the Project")
    if agree:
        st.title("About")

        st.header("Project Overview")
        st.write("""
        This project aims to predict human activity from videos using machine learning techniques.
        We utilize a convolutional LSTM model to recognize various activities such as Biking, Diving, Golf Swing, and Pizza Tossing.
        The model is trained on a dataset of labeled videos and can predict the activity shown in a given video clip.
        """)
        st.header("Examples")
        a=st.toggle("Biking")
        if a:
            st.image('C:/Users/veeri/Downloads/Biking.jpg', caption='Biking')
        b=st.toggle("Diving")
        if b:
            st.image('C:/Users/veeri/Downloads/Diving.jpg', caption='Diving')
        c=st.toggle("Golf Swing")
        if c:
            st.image('C:/Users/veeri/Downloads/Golf.jpg', caption='Golf Swing')
        d=st.toggle("Pizza Tossing")
        if d:
            st.image('C:/Users/veeri/Downloads/Pizza.jpg', caption='Pizza Tossing')

    add_selectbox = st.sidebar.selectbox("How would you like to upload a video?", ("YouTube URL", "Local Device", "Live Camera"))
    if add_selectbox == "YouTube URL":
        youtube_url = st.text_input("Enter the YouTube URL:")
        if youtube_url:
            st.write("You entered:", youtube_url)
            if st.button("Download YouTube Video"):
                test_videos_directory = 'test_videos'
                os.makedirs(test_videos_directory, exist_ok=True)
                video_file_path, video_title = download_youtube_video(youtube_url, test_videos_directory)
                st.success(f"Download complete! Video saved as: {video_title}.mp4")

                output_video_file_path = f'{test_videos_directory}/{video_title}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
                output_video_file_path = perform_action_recognition(video_file_path, output_video_file_path)

                st.success("Prediction complete! You can download the output video below.")
                st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)
                
    elif add_selectbox == "Local Device":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4'])
        if uploaded_file is not None:
            test_videos_directory = 'test_videos'
            os.makedirs(test_videos_directory, exist_ok=True)
            video_file_path = os.path.join(test_videos_directory, uploaded_file.name)
            with open(video_file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"You uploaded: {uploaded_file.name}")

            output_video_file_path = f'{test_videos_directory}/{uploaded_file.name}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
            output_video_file_path = perform_action_recognition(video_file_path, output_video_file_path)

            st.success("Prediction complete! You can download the output video below.")
            st.markdown(get_binary_file_downloader_html(output_video_file_path, "Download Predicted Video"), unsafe_allow_html=True)

    elif add_selectbox == "Live Camera":
        st.write("Click the button below to start the camera and perform activity detection.")
        if st.button("Predict from Camera"):
            cap = cv2.VideoCapture(0)  # Open the default camera (0) on your laptop
            frames_queue = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = perform_action_recognition_from_camera(frame, frames_queue)

                cv2.imshow('Human Activity Prediction', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
