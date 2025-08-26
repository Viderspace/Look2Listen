# av_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from avspeech.model.audio_model import AudioDilatedCNN
from avspeech.model.visual_model import VisualDilatedCNN, upsample_visual_features
LEAKY_SLOPE = 0.2


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=LEAKY_SLOPE, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# def init_weights(m):
#     # He for ReLU layers; Xavier for the final linear head
#     if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#         if isinstance(m, nn.Linear) and getattr(m, "_is_output_head", False):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         else:
#             nn.init.kaiming_normal_(m.weight, a=LEAKY_SLOPE, nonlinearity='leaky_relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.LSTM):
#         # Optional: set forget-gate bias = 1 for stability
#         for name, p in m.named_parameters():
#             if 'bias' in name:
#                 hidden = m.hidden_size
#                 p.data[hidden:2*hidden].fill_(1.0)

class AudioVisualModel(nn.Module):

    def __init__(self, audio_channels=2, video_embed_dim=512):
        """
        Single speaker enhancement model (no speaker separation)

        Args:
            audio_channels: 2 for complex STFT (real + imaginary)
            video_embed_dim: 512 for InceptionResnetV1 embeddings
        """
        super(AudioVisualModel, self).__init__()

        # Audio and Visual streams
        self.audio_cnn = AudioDilatedCNN()
        self.visual_cnn = VisualDilatedCNN(input_dim=video_embed_dim)

        # Calculate fusion dimensions
        audio_feature_dim = 8 * 257  # 8 channels × 257 freq bins = 2056
        visual_feature_dim = 256
        fused_dim = audio_feature_dim + visual_feature_dim  # 2312

        # Bidirectional LSTM
        self.blstm = nn.LSTM(
                input_size=fused_dim,
                hidden_size=400,
                num_layers=1,
                batch_first=True,
                bidirectional=True
        )

        # BiLSTM output will be 400*2 = 800 dimensions  # Sum-merged to 400-D in forward()

        # Three FC layers + output (added third hidden; removed FC BN; keep linear output)
        self.fc1 = nn.Linear(400, 600)  # input dim changed due to sum-merge
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)  # added hidden layer to match original 3×600
        self.fc4 = nn.Linear(600, 257 * 2)  # 257 freq × 2 (real/imag) – output layer (linear)

        self.bn1 = nn.BatchNorm1d(298)
        self.bn2 = nn.BatchNorm1d(298)
        self.bn3 = nn.BatchNorm1d(298)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)

        # Mark final head for Xavier init
        self.fc4._is_output_head = True

        # Apply initializers (He for ReLU stacks; Xavier for head; LSTM forget-bias tweak)
        self.apply(init_weights)

    def forward(self, audio_input, visual_input):
        """
        Args:
            audio_input: [batch, 257, 298, 2] - Complex STFT (your format)
            visual_input: [batch, 75, 512] - Face embeddings at 25fps
        """
        batch_size = audio_input.size(0)

        # Process both streams
        audio_features = self.audio_cnn(audio_input)
        visual_features = self.visual_cnn(visual_input)  # [batch, 256, 75]

        # Upsample visual features
        visual_features = upsample_visual_features(visual_features)  # [batch, 256, 298]

        # Reshape audio features for fusion
        audio_features = audio_features.permute(0, 3, 1, 2)  # [batch, 298, 8, 257]
        audio_features = audio_features.reshape(batch_size, 298, -1)  # [batch, 298, 2056]

        # Reshape visual features
        visual_features = visual_features.transpose(1, 2)  # [batch, 298, 256]

        # Concatenate along feature dim
        fused_features = torch.cat([audio_features, visual_features], dim=2)  # [batch, 298, 2312]

        # BiLSTM
        lstm_out, _ = self.blstm(fused_features)  # [batch, 298, 800]
        # Sum-merge directions → [batch, 298, 400]
        fwd, bwd = torch.chunk(lstm_out, 2, dim=2)
        x = fwd + bwd  # [batch, 298, 400]

        # ---- FC blocks with Leaky → BN(time) → Dropout ----
        # BN1d expects (N, C, L). Here we use C = time = 298, L = feature.
        x = F.leaky_relu(self.fc1(x), negative_slope=LEAKY_SLOPE)  # [B, 298, 600]
        # x = self.drop1(x)

        x = F.leaky_relu(self.fc2(x), negative_slope=LEAKY_SLOPE)  # [B, 298, 600]
        # x = self.drop2(x)

        x = F.leaky_relu(self.fc3(x), negative_slope=LEAKY_SLOPE)  # [B, 298, 600]
        # x = self.drop3(x)

        # Output head
        x = self.fc4(x)  # [B, 298, 514]
        x = torch.tanh(x)  # [-1, 1] bounded mask

        # Reshape back to [batch, 257, 298, 2]
        x = x.reshape(batch_size, 298, 257, 2)
        x = x.permute(0, 2, 1, 3)
        return x


