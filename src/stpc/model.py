# src/stpc/model.py
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A basic building block: two 1D convolutions + BatchNorm + ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    """
    1D U-Net for denoising. Now supports multi-channel input.
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet1D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.encoder.append(ConvBlock(current_channels, feature))
            current_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
    def encode(self, x):
        """
        Runs the input through the encoder and returns the bottleneck feature representation.
        """
        for down in self.encoder:
            x = down(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        embedding = torch.mean(x, dim=-1)
        return embedding

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2], mode='linear', align_corners=False
                )
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](concat_skip)

        return self.final_conv(x)

class ECGClassifier(nn.Module):
    """
    A simple 1D CNN for classifying individual heartbeats.
    """
    def __init__(self, num_classes=5):
        super(ECGClassifier, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc_block = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_block(x)
        return x