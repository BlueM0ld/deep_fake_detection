import streamlit as st
import requests
import psycopg2
import uuid
import tempfile
import cv2
from PIL import Image
import subprocess

def compress_video(input_path, output_path, target_bitrate="1000k"):
    """
    Compress the video using ffmpeg to reduce file size.
    """
    try:
        # Run ffmpeg command to compress video
        subprocess.run([
            "ffmpeg", "-i", input_path, "-b:v", target_bitrate, "-vcodec", "libx264",
            "-acodec", "aac", "-strict", "experimental", output_path
        ], check=True)
        st.success("Video compressed successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Error during video compression: {str(e)}")
        
st.set_page_config(page_title="Deepfake Video Classifier", layout="wide")

# Setup database connection
def setup_db_connection():
    try:
        return psycopg2.connect(
            "dbname=db user=postgres password=docker host=db"
        )
    except psycopg2.Error as e:
        st.error(f"Error connecting to the database: {e}")
        return None

# Insert classification result into the database
def insert_document(query, session_id, related_docs, selected_doc):
    try:
        connection = setup_db_connection()
        sql_executor = connection.cursor()

        insert_doc_query = """
            INSERT INTO user_logs (session_id, search_query, related_docs, selected_doc)
            VALUES (%s, %s, %s, %s)
        """

        sql_executor.execute(
            insert_doc_query, (session_id, query, related_docs, selected_doc)
        )

        # Persist database changes
        connection.commit()

        sql_executor.close()
        connection.close()
        st.success("Classification results stored in the database.")

    except Exception as e:
        st.error(f"Error in inserting documents: {e}")

# Streamlit app layout
st.title("Deepfake Video Classifier")

col1, col2 = st.columns(2)

with col1:
    # Update file uploader to accept videos
    uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

classify_button = st.button("Classify Video")

# Initialize session state
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Handle video classification when button is pressed and file is uploaded
if classify_button and uploaded_file:
    try:
        # Save uploaded video to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.getvalue())
            video_path = temp_video.name

        # Process the video, compress it after classification
        compressed_video_path = video_path.replace(".mp4", "_compressed.mp4")

        # Compress the video (you can use the compress_video function as described earlier)
        compress_video(video_path, compressed_video_path)

        # Send video to FastAPI for classification (optional if needed)
        with open(compressed_video_path, "rb") as video_file:
            files = {'file': (uploaded_file.name, video_file, 'video/mp4')}

            # Send video to FastAPI for classification
            response = requests.post(
                "http://deep_fake_detection-fastapi-1:8000/classify",
                files=files
            )

        if response.status_code == 200:
            st.session_state.classification_result = response.json()["class"]

            # Use columns to control layout and display the video
            with col1:
                st.video(compressed_video_path)

            with col2:
                st.subheader("Classification Result:")
                st.write(f"The video is classified as: **{st.session_state.classification_result}**")

            # Store the result in the database
            insert_document(
                query="deepfake_classification",
                session_id=st.session_state.session_id,
                related_docs=[st.session_state.classification_result],
                selected_doc=0
            )
        else:
            st.error(f"Error: Unable to classify video. Status code: {response.status_code}")
            st.write(f"Response content: {response.text}")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

else:
    st.write("Upload a video and click 'Classify Video' to see results.")
