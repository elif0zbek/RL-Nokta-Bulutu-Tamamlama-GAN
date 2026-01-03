import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, latent_dim, 1); self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        # Input: (B, N, 3) -> Conv1d bekler: (B, 3, N)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, 1024)

class PointNetGenerator(nn.Module):
    def __init__(self, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.encoder = PointNetEncoder()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3)

    def forward(self, x):
        feat = self.encoder(x)
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.num_points, 3)

class PointNetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.encoder(x)
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))