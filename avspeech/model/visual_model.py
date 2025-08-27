
import torch.nn as nn
import torch.nn.functional as F
LEAKY_SLOPE = 0.2

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=LEAKY_SLOPE, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class VisualDilatedCNN(nn.Module):

    def __init__(self, input_dim=512):
        super(VisualDilatedCNN, self).__init__()

        # Based on Table 2 - treating temporal dimension only
        # Format: (out_channels, kernel_size, dilation, padding)  # padding hard-coded
        layer_specs = [
            (256, 7,  1,  3),   # conv1: k=7,  d=1  -> pad=3
            (256, 5,  1,  2),   # conv2: k=5,  d=1  -> pad=2
            (256, 5,  2,  4),   # conv3: k=5,  d=2  -> pad=4
            (256, 5,  4,  8),   # conv4: k=5,  d=4  -> pad=8
            (256, 5,  8, 16),   # conv5: k=5,  d=8  -> pad=16
            (256, 5, 16, 32),   # conv6: k=5,  d=16 -> pad=32
        ]

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_channels = input_dim

        for out_channels, kernel_size, dilation, padding in layer_specs:
            # Conv1d for temporal processing (SAME length via hard-coded padding)
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False
            )
            self.conv_layers.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        # Apply He init to temporal conv stack (Leaky-aware)
        self.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: [batch, 75, 512] face embeddings
        Returns:
            [batch, 256, 75] visual features
        """
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # [batch, 512, 75]

        assert x.shape == (x.shape[0], 512, 75), f"Expected input shape [batch, 512, 75], got {x.shape}"

        # Apply convolutions
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)

        return x


def upsample_visual_features(x, target_length=298):
    return F.interpolate(x, size=target_length, mode='nearest')

# def upsample_visual_features(visual_features, target_length=298):
#     """
#     Upsample visual features from 25Hz to 100Hz using nearest neighbor
#
#     Args:
#         visual_features: [batch, 256, 75] at 25Hz
#         target_length: 298 (for 3 seconds at 100Hz)
#
#     Returns:
#         [batch, 256, 298] upsampled features
#     """
#     # Better version - no unnecessary unpacking
#     visual_features = visual_features.unsqueeze(-1)
#     upsampled = F.interpolate(visual_features, size=(target_length, 1), mode='nearest')
#     return upsampled.squeeze(-1)

