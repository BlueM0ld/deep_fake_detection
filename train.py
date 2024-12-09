from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from model.encoder import DeepFakeEncoder
from video_dataset import VideoDataset
from torchvision import transforms
import wandb

torch.cuda.empty_cache()

# Hyperparameters and configurations
DIMENSIONS = 32
NUM_OF_LAYERS = 2
FF_DIMENSIONS = 32
MAX_SEQ_LEN = 300
NUM_OF_HEADS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 1
NUM_OF_EPOCHS = 1

# Initialize model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DeepFakeEncoder(
    dim=DIMENSIONS,
    num_layers=NUM_OF_LAYERS,
    dim_ff=FF_DIMENSIONS,
    max_seq_len=MAX_SEQ_LEN,
    num_heads=NUM_OF_HEADS,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize wandb
wandb.init(
    project="deepfake",
    config={
        "dim_model": DIMENSIONS,
        "dim_ff": FF_DIMENSIONS,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_OF_EPOCHS,
        "batch_size": BATCH_SIZE,
    },
)

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_dir = "dfdc_train_part_0"
metadata_file = "dfdc_train_part_0/metadata.json"

dataset = VideoDataset(video_dir=video_dir, metadata_file=metadata_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# Training loop
for epoch in range(NUM_OF_EPOCHS):
    model.train()
    total_loss = 0
    total_accuracy = 0
    n_batches = len(dataloader)
    
    for batch_idx, (video_frames, labels) in enumerate(dataloader):
        video_frames, labels = video_frames.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(video_frames)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean().item()
        total_accuracy += accuracy

        # Log batch metrics
        wandb.log({
            "batch_loss": loss.item(),
            "batch_accuracy": accuracy,
            "batch_idx": batch_idx + epoch * n_batches,
        })

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_OF_EPOCHS}], Batch [{batch_idx}/{n_batches}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
    
    # Log epoch metrics
    avg_loss = total_loss / n_batches
    avg_accuracy = total_accuracy / n_batches
    print(f"Epoch [{epoch+1}/{NUM_OF_EPOCHS}], Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
    wandb.log({"epoch_loss": avg_loss, "epoch_accuracy": avg_accuracy})

# Save model
torch.save(model.state_dict(), "deepfake_encoder.pth")
wandb.save("deepfake_encoder.pth")

# Evaluation
model.eval()
test_accuracy = 0
test_count = 0
test_predictions = []

with torch.no_grad():
    for video_frames, labels in dataloader:
        video_frames, labels = video_frames.to(device), labels.to(device)
        outputs = model(video_frames)
        _, predicted = torch.max(outputs, 1)
        
        test_accuracy += (predicted == labels).float().sum().item()
        test_count += labels.size(0)
        test_predictions.extend(predicted.cpu().numpy())

# Calculate and log test accuracy
final_accuracy = test_accuracy / test_count
print(f"Test Accuracy: {final_accuracy * 100:.2f}%")
wandb.log({"final_test_accuracy": final_accuracy})

wandb.finish()
