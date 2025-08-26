import torch.nn as nn
import torch.nn.functional as F

LEAKY_SLOPE = 0.2
def _calculate_same_padding(kernel_size, dilation):
    """Calculate padding to maintain input dimensions (TensorFlow 'SAME' padding)"""
    # For 'SAME' padding: padding = (kernel_size - 1) * dilation / 2
    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=LEAKY_SLOPE, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class AudioDilatedCNN(nn.Module):

    def __init__(self):
        super(AudioDilatedCNN, self).__init__()

        # Layer specifications from Table 1 (keeping permute -> [B, C, F, T])
        # NOTE: To keep time-growth while using H=freq, W=time, we:
        #   1) swap the first two kernels (7x1) <-> (1x7)
        #   2) flip early dilations (dH,dW): (2,1)->(1,2), (4,1)->(1,4), ...
        # Pads are hard-coded for SAME output size.
        # Format: (out_channels, kernel_size, dilation, padding)
        layer_specs = [
                # Layer 1-2: Initial spatial processing (kernels swapped)
                (96, (7, 1), (1, 1), (3, 0)),  # conv1: 7x1 over F (freq)
                (96, (1, 7), (1, 1), (0, 3)),  # conv2: 1x7 over T (time)

                # Layer 3-8: Time-dilated convolutions (grow along time = W)
                (96, (5, 5), (1, 1), (2, 2)),  # conv3
                (96, (5, 5), (1, 2), (2, 4)),  # conv4  (was 2x1)
                (96, (5, 5), (1, 4), (2, 8)),  # conv5  (was 4x1)
                (96, (5, 5), (1, 8), (2, 16)),  # conv6  (was 8x1)
                (96, (5, 5), (1, 16), (2, 32)),  # conv7  (was 16x1)
                (96, (5, 5), (1, 32), (2, 64)),  # conv8  (was 32x1)

                # Layer 9-14: Time-frequency dilated convolutions (isotropic kept)
                (96, (5, 5), (1, 1), (2, 2)),  # conv9
                (96, (5, 5), (2, 2), (4, 4)),  # conv10
                (96, (5, 5), (4, 4), (8, 8)),  # conv11
                (96, (5, 5), (8, 8), (16, 16)),  # conv12
                (96, (5, 5), (16, 16), (32, 32)),  # conv13
                (96, (5, 5), (32, 32), (64, 64)),  # conv14

                # Layer 15: Output projection
                (8, (1, 1), (1, 1), (0, 0)),  # conv15
        ]

        # Build the layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_channels = 2  # Input: real and imaginary parts of STFT

        for i, (out_channels, kernel_size, dilation, padding) in enumerate(layer_specs):
            if i < len(layer_specs) - 1:
                print(f"Layer {i + 1}: kernel={kernel_size}, dilation={dilation}, padding={padding}")
            else:
                # last layer already has (0,0) padding in spec; keep the log consistent
                print(f"Layer {i + 1}: kernel={kernel_size}, dilation={dilation}, padding={padding}")

            conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    bias=False  # Bias is handled by BatchNorm
            )

            self.conv_layers.append(conv)

            # Batch normalization after all but the last conv
            if i < len(layer_specs) - 1:
                self.batch_norms.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels

        # Final BatchNorm for the last conv (Conv15) to match original TF tail
        self.final_bn = nn.BatchNorm2d(layer_specs[-1][0])  # out_channels of last layer (8)

        # Apply He init to conv layers (ReLU activations throughout)
        self.apply(init_weights)


    def forward(self, x):
        """
        Args:
            x: Input spectrogram tensor of shape [batch, 2, freq_bins, time_frames]
               where 2 channels are real and imaginary parts
        Returns:
            Audio features of shape [batch, 8, freq_bins, time_frames]
        """
        # Transpose to Conv2d format: [batch, freq, time, channels] -> [batch, channels, freq, time]
        # print(f"Input shape: {x.shape}")
        x = x.permute(0, 3, 1, 2)  # [batch, 2, 257, 298]
        # print(f"Input shape after permute: {x.shape}")

        # Process through all conv layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)

        # Last layer + BatchNorm + ReLU (to match original TF implementation)
        x = self.conv_layers[-1](x)
        x = self.final_bn(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)

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

