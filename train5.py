import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model.encoder import DeepFakeEncoder
from video_dataset import VideoDataset
from torchvision import transforms
import wandb
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.metrics import f1_score 
from collections import Counter
import os

torch.cuda.empty_cache()

# Constants
DIMENSIONS = 64
NUM_OF_LAYERS = 4
FF_DIMENSIONS = 64
MAX_SEQ_LEN = 300
NUM_OF_HEADS = 4
LEARNING_RATE = 0.0001
NUM_OF_EPOCHS = 3
BATCH_SIZE = 6
CHECKPOINT_PATH = "deepfake_encoder_checkpoint.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Initialization
model = DeepFakeEncoder(
    dim=DIMENSIONS,
    num_layers=NUM_OF_LAYERS,
    dim_ff=FF_DIMENSIONS,
    max_seq_len=MAX_SEQ_LEN,
    num_heads=NUM_OF_HEADS,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# WandB Initialization
wandb.init(
    project="deepfake",
    config={
        "architecture": "DeepFakeEncoder",
        "dim_model": DIMENSIONS,
        "dim_ff": FF_DIMENSIONS,
        "num_layers": NUM_OF_LAYERS,
        "max_seq_len": MAX_SEQ_LEN,
        "num_heads": NUM_OF_HEADS,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_OF_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
    },
)

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_dir = "dfdc_train_part_01"
metadata_file = "dfdc_train_part_01/metadata.json"

# Dataset and DataLoader
dataset = VideoDataset(video_dir=video_dir, metadata_file=metadata_file, transform=transform)
print("Total number of samples in dataset:", len(dataset))

# Check label counts
print("Label counts:", dataset.label_counts)

# Use the precomputed label counts from dataset.label_counts
label_to_weight = {label: 1.0 / count for label, count in dataset.label_counts.items()}
sample_weights = [label_to_weight[label] for label in dataset.label_counts.keys()]

# Initialize the WeightedRandomSampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

# Dataset and DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

# Subset creation for training and validation
subset = torch.utils.data.Subset(dataset, list(range(1000)))
print("Length of the subset dataset:", len(subset))

# 80% train, 20% validation split
val_split = int(len(subset) * 0.2)
train_subset, val_subset = torch.utils.data.random_split(subset, [len(subset) - val_split, val_split])

val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE)

# Load checkpoint if available
if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint from:", CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0  # Start fresh training if no checkpoint exists

# Training loop
model.train()
for epoch in range(start_epoch, NUM_OF_EPOCHS):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    all_train_labels = []
    all_train_predictions = []
    batch_count = 0

    for batch_idx, (video_frames, labels) in enumerate(dataloader):
        video_frames, labels = video_frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(video_frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean().item()

        # Collect predictions and labels for F1
        all_train_labels.extend(labels.cpu().numpy())
        all_train_predictions.extend(predicted.cpu().numpy())

        # Update epoch statistics
        epoch_loss += loss.item()
        epoch_accuracy += accuracy
        batch_count += 1

        # Log batch metrics
        wandb.log({
            "batch/loss": loss.item(),
            "batch/accuracy": accuracy,
            "batch/learning_rate": optimizer.param_groups[0]['lr']
        })

        # Print batch progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_OF_EPOCHS}], "
                  f"Batch [{batch_idx}/{len(dataloader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy:.4f}")

    # Calculate F1 Score
    train_f1_score = f1_score(all_train_labels, all_train_predictions, average='weighted')

    # Average epoch metrics
    epoch_loss /= batch_count
    epoch_accuracy /= batch_count

    # Log epoch metrics
    wandb.log({
        "epoch/loss": epoch_loss,
        "epoch/accuracy": epoch_accuracy,
        "epoch/f1_score": train_f1_score,
        "epoch": epoch
    })
    print(f"Epoch [{epoch+1}/{NUM_OF_EPOCHS}] completed. "
          f"Average Loss: {epoch_loss:.4f}, "
          f"Average Accuracy: {epoch_accuracy:.4f}, "
          f"F1 Score: {train_f1_score:.4f}")

    # Save model at the end of each epoch (overwrite to save space)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_PATH)

# Validation
print("Evaluating")
model.eval()
val_loss = 0.0
val_accuracy = 0.0
val_batches = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for val_batch_idx, (val_video_frames, val_labels) in enumerate(val_dataloader):
        val_video_frames, val_labels = val_video_frames.to(device), val_labels.to(device)
        val_outputs = model(val_video_frames)

        # Calculate loss and accuracy for the current batch
        val_loss_batch = criterion(val_outputs, val_labels)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy_batch = (val_predicted == val_labels).float().mean().item()

        # Accumulate metrics
        val_loss += val_loss_batch.item()
        val_accuracy += val_accuracy_batch
        val_batches += 1

        # Append data for F1 Score computation
        all_predictions.extend(val_predicted.cpu().numpy())
        all_labels.extend(val_labels.cpu().numpy())

        # Log batch-wise results
        wandb.log({
            "validation/batch_loss": val_loss_batch.item(),
            "validation/batch_accuracy": val_accuracy_batch,
            "validation/batch_index": val_batch_idx
        })

        # Print intermediate results
        print(f"Validation Batch [{val_batch_idx}/{len(val_dataloader)}]: "
              f"Loss = {val_loss_batch.item():.4f}, Accuracy = {val_accuracy_batch:.4f}")

# Calculate F1 Score for validation
val_f1_score = f1_score(all_labels, all_predictions, average='weighted')

# Final average validation metrics
val_loss /= val_batches
val_accuracy /= val_batches

print(f"Validation Completed: Average Loss = {val_loss:.4f}, "
      f"Average Accuracy = {val_accuracy:.4f}, "
      f"F1 Score = {val_f1_score:.4f}")

# Save Model (final checkpoint)
torch.save({
    'epoch': NUM_OF_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
}, CHECKPOINT_PATH)

wandb.log({
    "validation/average_loss": val_loss,
    "validation/average_accuracy": val_accuracy,
    "validation/f1_score": val_f1_score
})

wandb.finish()
