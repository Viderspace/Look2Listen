import torch.nn as nn
import torch.nn.functional as F


def _calculate_same_padding(kernel_size, dilation):
    """Calculate padding to maintain input dimensions (TensorFlow 'SAME' padding)"""
    # For 'SAME' padding: padding = (kernel_size - 1) * dilation / 2
    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


class AudioDilatedCNN(nn.Module):

    def __init__(self):
        super(AudioDilatedCNN, self).__init__()

        # Layer specifications from Table 1 in the paper
        # Format: (num_filters, kernel_size, dilation)
        layer_specs = [
                # Layer 1-2: Initial spatial processing
                (96, (1, 7), (1, 1)),  # conv1: 1x7 kernel
                (96, (7, 1), (1, 1)),  # conv2: 7x1 kernel

                # Layer 3-8: Time-dilated convolutions (expanding receptive field in time)
                (96, (5, 5), (1, 1)),  # conv3: 5x5, dilation 1x1
                (96, (5, 5), (2, 1)),  # conv4: 5x5, dilation 2x1
                (96, (5, 5), (4, 1)),  # conv5: 5x5, dilation 4x1
                (96, (5, 5), (8, 1)),  # conv6: 5x5, dilation 8x1
                (96, (5, 5), (16, 1)),  # conv7: 5x5, dilation 16x1
                (96, (5, 5), (32, 1)),  # conv8: 5x5, dilation 32x1

                # Layer 9-14: Time-frequency dilated convolutions
                (96, (5, 5), (1, 1)),  # conv9: 5x5, dilation 1x1
                (96, (5, 5), (2, 2)),  # conv10: 5x5, dilation 2x2
                (96, (5, 5), (4, 4)),  # conv11: 5x5, dilation 4x4
                (96, (5, 5), (8, 8)),  # conv12: 5x5, dilation 8x8
                (96, (5, 5), (16, 16)),  # conv13: 5x5, dilation 16x16
                (96, (5, 5), (32, 32)),  # conv14: 5x5, dilation 32x32

                # Layer 15: Output projection
                (8, (1, 1), (1, 1))  # conv15: 1x1, 8 filters (output channels)
        ]

        # Build the layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_channels = 2  # Input: real and imaginary parts of STFT

        for i, (out_channels, kernel_size, dilation) in enumerate(layer_specs):
            # Calculate padding to maintain spatial dimensions (equivalent to 'SAME' padding)
            if i < len(layer_specs) - 1:  # All layers except the last
                padding = _calculate_same_padding(kernel_size, dilation)
            else:  # Last layer (1x1 conv) doesn't need padding
                padding = (0, 0)

            conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    bias=False  # Bias is handled by BatchNorm
            )

            self.conv_layers.append(conv)

            # Batch normalization for all layers except the last
            if i < len(layer_specs) - 1:
                self.batch_norms.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels


    def forward(self, x):
        """
        Args:
            x: Input spectrogram tensor of shape [batch, 2, freq_bins, time_frames]
               where 2 channels are real and imaginary parts
        Returns:
            Audio features of shape [batch, 8, freq_bins, time_frames]
        """
        # Transpose to Conv2d format: [batch, freq, time, channels] -> [batch, channels, freq, time]
        x = x.permute(0, 3, 1, 2)  # [batch, 2, 257, 298]

        # Process through all conv layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        # Last layer (no batch norm or activation)
        x = self.conv_layers[-1](x)

        return x


    def freeze_early_layers(self, num_layers: int = 2):
        """Freeze first N convolutional layers"""
        for i in range(min(num_layers, len(self.conv_layers))):
            for param in self.conv_layers[i].parameters():
                param.requires_grad = False
            if i < len(self.batch_norms):
                for param in self.batch_norms[i].parameters():
                    param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True


def calculate_receptive_field(layer_specs):
    """Calculate the receptive field after each layer"""
    rf_h, rf_w = 1, 1  # Initial receptive field

    for i, (_, kernel_size, dilation) in enumerate(layer_specs):
        # Receptive field grows by: (kernel_size - 1) * dilation
        rf_h += (kernel_size[0] - 1) * dilation[0]
        rf_w += (kernel_size[1] - 1) * dilation[1]

        print(f"Layer {i + 1}: Receptive field = {rf_h}×{rf_w}")