import torch
import torch.optim as optim
import torch.nn as nn
from model.encoder import DeepFakeEncoder
from video_dataset import VideoDataset
from torchvision import transforms
import wandb
import numpy as np

torch.cuda.empty_cache()

# Constants
DIMENSIONS = 32
NUM_OF_LAYERS = 2
FF_DIMENSIONS = 32
MAX_SEQ_LEN = 300
NUM_OF_HEADS = 2
LEARNING_RATE = 0.001
NUM_OF_EPOCHS = 10
BATCH_SIZE = 3

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

# Initialize wandb with more detailed config
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

# Watch the model to track gradients and parameters
wandb.watch(model, criterion, log="all", log_freq=10)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_dir = "dfdc_train_part_0"
metadata_file = "dfdc_train_part_0/metadata.json"

dataset = VideoDataset(video_dir=video_dir, metadata_file=metadata_file, transform=transform)
subset = torch.utils.data.Subset(dataset, list(range(500)))
dataloader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE)

def log_model_statistics():
    """Log model weight and bias statistics"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            wandb.log({
                f"params/{name}_mean": param.data.mean().item(),
                f"params/{name}_std": param.data.std().item(),
                f"grads/{name}_mean": param.grad.mean().item() if param.grad is not None else 0,
                f"grads/{name}_std": param.grad.std().item() if param.grad is not None else 0
            })

def log_layer_activations(outputs):
    """Log activation statistics"""
    wandb.log({
        "activations/output_mean": outputs.mean().item(),
        "activations/output_std": outputs.std().item(),
        "activations/output_min": outputs.min().item(),
        "activations/output_max": outputs.max().item()
    })

# Training loop with enhanced logging
model.train()
running_loss = []
running_accuracy = []

for epoch in range(NUM_OF_EPOCHS):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
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
        
        # Update running statistics
        epoch_loss += loss.item()
        epoch_accuracy += accuracy
        batch_count += 1
        
        # Log batch metrics
        wandb.log({
            "batch/loss": loss.item(),
            "batch/accuracy": accuracy,
            "batch/learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Log model statistics every 10 batches
        if batch_idx % 10 == 0:
            log_model_statistics()
            log_layer_activations(outputs)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_OF_EPOCHS}], "
                  f"Batch [{batch_idx}/{len(dataloader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy:.4f}")
    
    # Calculate and log epoch metrics
    epoch_loss /= batch_count
    epoch_accuracy /= batch_count
    
    wandb.log({
        "epoch/loss": epoch_loss,
        "epoch/accuracy": epoch_accuracy,
        "epoch": epoch
    })
    
    print(f"Epoch [{epoch+1}/{NUM_OF_EPOCHS}] completed. "
          f"Average Loss: {epoch_loss:.4f}, "
          f"Average Accuracy: {epoch_accuracy:.4f}")

# Save model and finish wandb run
torch.save({
    'epoch': NUM_OF_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, "deepfake_encoder_checkpoint.pth")

wandb.finish()