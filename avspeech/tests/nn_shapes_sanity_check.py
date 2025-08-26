#!/usr/bin/env python3
"""
simple_run.py — minimal hard-coded runner for your audio CNN.
- Imports `AudioDilatedCNN` from `audio_model.py` in the same folder.
- Feeds a dummy batch and prints input/output shapes.
- Add your own print()s inside the model's forward() to see layer-wise shapes.
"""
import torch
from avspeech.model.audio_model import AudioDilatedCNN  # change the class name if yours differs

# ---- Hard-coded shapes ----
BATCH = 1
FREQ  = 257
TIME  = 298
CH    = 2

# Toggle this if your model expects [B, C, F, T] instead.
CHANNELS_LAST = True  # True -> [B, F, T, C]; False -> [B, C, F, T]

def main():
    model = AudioDilatedCNN()  # no-arg init; adjust if your ctor needs args
    model.eval()

    if CHANNELS_LAST:
        x = torch.randn(BATCH, FREQ, TIME, CH)  # [B, F, T, C]
    else:
        x = torch.randn(BATCH, CH, FREQ, TIME)  # [B, C, F, T]

    print("Input shape:", tuple(x.shape))
    with torch.no_grad():
        y = model(x)
    print("Output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
