import streamlit as st
import requests

# ... existing imports ...

def upload_video(video_file):
    """Uploads the video file to the backend."""
    if video_file is not None:
        # Assuming you have an endpoint to handle video uploads
        response = requests.post("http://localhost:8000/upload_video", files={"file": video_file})
        if response.status_code == 200:
            st.success("Video uploaded successfully!")
        else:
            st.error("Failed to upload video.")

def main():
    st.title("Deep Fake Detection App")
    
    # Video upload section
    st.header("Upload Video")
    video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if st.button("Upload Video"):
        upload_video(video_file)

    # Display uploaded video
    if video_file is not None:
        st.video(video_file)

# Run the app
if __name__ == "__main__":
    main() 