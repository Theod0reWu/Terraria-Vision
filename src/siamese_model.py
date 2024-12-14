import torch
import torch.nn as nn

class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 1 * 1, embedding_dim)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.AdaptiveAvgPool2d(1)(x)  # Average pooling for fixed-size output
        x = x.view(x.size(0), -1)
        return nn.functional.normalize(self.fc(x), p=2, dim=1)  # L2 normalized embedding

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        distances = torch.norm(embedding1 - embedding2, p=2, dim=1)
        loss_similar = label * distances**2
        loss_dissimilar = (1 - label) * torch.clamp(self.margin - distances, min=0)**2
        return torch.mean(loss_similar + loss_dissimilar) / 2
