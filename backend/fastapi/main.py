import io
import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model.encoder import DeepFakeEncoder
from torchvision import transforms
from pathlib import Path
from PIL import Image
import cv2
import tempfile

# Initialize global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    # Load the deepfake model
    model = DeepFakeEncoder(
        dim=48,  # Example values
        num_layers=4,
        dim_ff=48,
        max_seq_len=300,
        num_heads=4,
    ).to(device)

    # Load model weights
    checkpoint = torch.load('deepfake_encoder_checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    yield
    
    # Clean up resources
    model = None

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_video(file_path):
    """
    Load the video using OpenCV and preprocess frames.
    """
    cap = cv2.VideoCapture(file_path)
    frames = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Apply transformation to each frame
            frame = transform(Image.fromarray(frame))
            frames.append(frame)
    except Exception as e:
        print(f"Error loading video {file_path}: {e}")
    finally:
        cap.release()
    return torch.stack(frames)

@app.post("/classify")
async def classify_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            video_path = temp_video.name

        # Load and preprocess video frames
        video_frames = load_video(video_path).to(device)

        # Classify using the model
        with torch.no_grad():
            outputs = model(video_frames.unsqueeze(0))  # Add batch dimension
            _, predicted_class = torch.max(outputs, 1)

        # Clean up temporary file
        os.remove(video_path)

        return {"class": "FAKE" if predicted_class.item() == 0 else "REAL"}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
