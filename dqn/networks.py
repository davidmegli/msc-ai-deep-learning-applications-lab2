'''
Author: David Megli
Date: 2025-05-02
File: networks.py
Description: QNetwork implementation
'''
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class CNNQNetwork(nn.Module):
    def __init__(self, input_channels, action_dim, hidden_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # [B, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # [B, 64, 9, 9]
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = x / 255.0  # Normalize image pixels to [0, 1]
        x = self.conv(x)
        x = self.fc(x)
        return x