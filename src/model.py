# src/model.py
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
    # MODIFICATION: Changed the __init__ signature to accept in_channels and out_channels
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet1D, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder
        # MODIFICATION: The first encoder block uses the provided `in_channels`
        current_channels = in_channels
        for feature in features:
            self.encoder.append(ConvBlock(current_channels, feature))
            current_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # The input to this ConvBlock is the concatenated skip connection
            # The number of channels is feature (from skip) + feature (from up-conv) = feature * 2
            self.decoder.append(ConvBlock(feature * 2, feature))

        # Final projection layer
        # MODIFICATION: The final layer projects to the desired number of `out_channels`
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
    def encode(self, x):
        """
        Runs the input through the encoder part of the U-Net and returns
        the feature representation from the bottleneck, pooled into a vector.
        """
        # Encoder path
        for down in self.encoder:
            x = down(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Global average pooling to get a fixed-size vector representation
        embedding = torch.mean(x, dim=-1)
        return embedding

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for correct order
        skip_connections = skip_connections[::-1]

        # Decoder path
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # up-conv
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2], mode='linear', align_corners=False
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](concat_skip)

        return self.final_conv(x)

# Verification script
if __name__ == "__main__":
    print("--- Verifying U-Net 1D Architecture ---")
    test_input = torch.randn(4, 1, 2048)  # batch=4, channels=1, length=2048
    model = UNet1D(in_channels=1, out_channels=1)
    output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    assert test_input.shape == output.shape, "Mismatch: Output length != Input length"
    print("✅ Verification successful: Temporal resolution preserved.")
