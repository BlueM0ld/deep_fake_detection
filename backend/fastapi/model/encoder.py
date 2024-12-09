import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.cnn import PretrainedCNN  
from model.positional import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, dim, dim_ff, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttention(dim, num_heads)
        self.norm_attn = nn.LayerNorm(dim)
        
        # Fully connected feed-forward network
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Self-attention layer
        x2, _ = self.self_attention(x, x, x, mask)
        x = self.norm_attn(x + self.dropout(x2))
        
        # Feed-forward network with residual connection
        x2 = self.fc2(torch.relu(self.fc1(x)))
        x = self.norm_ffn(x + self.dropout(x2))
        return x


class DeepFakeEncoder(nn.Module):
    def __init__(self, dim, num_layers, dim_ff, max_seq_len, num_heads):
        super(DeepFakeEncoder, self).__init__()
        
        # Use PretrainedCNN to extract features from frames
        self.cnn_backbone = PretrainedCNN(cnn_name='resnet50')  # Use your PretrainedCNN
        
        # Project CNN features to transformer embedding dimension
        self.projection = nn.Linear(self.cnn_backbone.cnn_features, dim)  # Use cnn_features from PretrainedCNN
        
        # Positional encoding to capture spatial and temporal relationships
        self.positional_encoding = PositionalEncoding(dim, max_seq_len)
        
        # Encoder layers for transformer
        self.layers = nn.ModuleList([EncoderLayer(dim, dim_ff, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

        # Classification head
        self.fc = nn.Linear(dim, 2)  # Binary classification

    def forward(self, frames, mask=None):
        batch_size, seq_len, _, height, width = frames.shape
        
        # Extract features using PretrainedCNN
        frames = frames.view(batch_size * seq_len, 3, height, width)  # Reshape frames for CNN
        cnn_features = self.cnn_backbone(frames)  # Shape: [batch_size * seq_len, 2048, 1, 1]
        cnn_features = cnn_features.view(batch_size, seq_len, self.cnn_backbone.cnn_features)  # Shape: [batch_size, seq_len, 2048]
        
        # Project CNN features to transformer embedding dimension
        x = self.projection(cnn_features)  # Shape: [batch_size, seq_len, dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)  # Shape: [batch_size, seq_len, dim]
        
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        
        # Aggregate features across frames (using the last frame)
        x = x[:, -1, :]  # Using the last frame's output for classification (can be changed)
        
        # Classification head
        logits = self.fc(x)  # Shape: [batch_size, num_classes]
        
        return logits  
