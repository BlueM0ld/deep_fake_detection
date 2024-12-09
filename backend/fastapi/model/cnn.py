import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedCNN(nn.Module):
    def __init__(self, cnn_name='resnet50'):
        super(PretrainedCNN, self).__init__()
        if cnn_name == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)  # Load ResNet50
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove final classification layer
        else:
            raise ValueError("Unsupported CNN model. Please use 'resnet50'.")

        self.cnn_features = 2048  # ResNet50 output size

    def forward(self, x):
        return self.cnn(x)  # Extract features
