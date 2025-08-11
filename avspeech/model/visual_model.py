
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualDilatedCNN(nn.Module):

    def __init__(self, input_dim=512):
        super(VisualDilatedCNN, self).__init__()

        # Project 512D to 1024D
        self.input_projection = nn.Linear(input_dim, 1024)

        # Based on Table 2 - treating temporal dimension only
        layer_specs = [
                (256, 7, 1),  # conv1
                (256, 5, 1),  # conv2
                (256, 5, 2),  # conv3
                (256, 5, 4),  # conv4
                (256, 5, 8),  # conv5
                (256, 5, 16),  # conv6
        ]

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_channels = 1024

        for out_channels, kernel_size, dilation in layer_specs:
            # Conv1d for temporal processing
            conv = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2,  # SAME padding
                    bias=False
            )
            self.conv_layers.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: [batch, 75, 512] face embeddings
        Returns:
            [batch, 256, 75] visual features
        """
        # Project embeddings
        x = self.input_projection(x)  # [batch, 75, 1024]

        # Transpose for Conv1d
        x = x.transpose(1, 2)  # [batch, 1024, 75]

        # Apply convolutions
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        # Output: [batch, 256, 75]
        return x


from timer_decorator import timer


def upsample_visual_features(visual_features, target_length=298):
    """
    Upsample visual features from 25Hz to 100Hz using nearest neighbor

    Args:
        visual_features: [batch, 256, 75] at 25Hz
        target_length: 298 (for 3 seconds at 100Hz)

    Returns:
        [batch, 256, 298] upsampled features
    """
    # Better version - no unnecessary unpacking
    visual_features = visual_features.unsqueeze(-1)
    upsampled = F.interpolate(
        visual_features,
        size=(target_length, 1),
        mode='nearest'
    )
    return upsampled.squeeze(-1)


# visual_cnn.py (alternative upsampling)
def upsample_visual_features_fast(visual_features, target_length=298):
    """
    Fast nearest neighbor upsampling from 75 to 298 frames
    """
    batch, channels, time = visual_features.shape  # [batch, 256, 75]

    # Calculate indices for nearest neighbor
    # 75 frames -> 298 frames (factor ~3.973)
    source_indices = torch.linspace(0, time - 1, target_length).long()

    # Direct indexing
    upsampled = visual_features[:, :, source_indices]

    return upsampled