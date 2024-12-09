import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class VideoDataset(Dataset):
    def __init__(self, video_dir, metadata_file, transform=None):
        """
        Args:
            video_dir (str): Path to the directory containing videos.
            metadata_file (str): Path to the metadata JSON file with labels.
            transform (callable, optional): Transformations to apply to video frames.
        """
        self.video_dir = video_dir
        self.transform = transform
        
        # Load metadata from JSON
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.video_filenames = list(self.metadata.keys())
        self.label_counts = self._count_labels()  # Precompute label counts

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, idx):
        video_filename = self.video_filenames[idx]
        video_path = os.path.join(self.video_dir, video_filename)
        
        # Load video frames
        video_frames = self._load_video(video_path)
        
        # Get label
        label = self.metadata[video_filename]["label"]
        label = 0 if label == "FAKE" else 1
        
        # Apply transformations (if any)
        if self.transform:
            video_frames = torch.stack([self.transform(Image.fromarray(frame)) for frame in video_frames])
        
        return video_frames, torch.tensor(label)

    def _load_video(self, video_path):
        """
        Load a video and extract frames. This function uses OpenCV to load the video.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
        finally:
            cap.release()
        
        return frames

    def _count_labels(self):
        """
        Count the number of FAKE and REAL labels in the dataset.
        """
        fake_count = 0
        real_count = 0
        for video_filename in self.video_filenames:
            label = self.metadata[video_filename]["label"]
            if label == "FAKE":
                fake_count += 1
            else:
                real_count += 1
        return {"FAKE": fake_count, "REAL": real_count}
