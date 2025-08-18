# from __future__ import annotations
# from pathlib import Path
# import random
# from typing import Optional, Iterator, List


# class ShuffledWavIterator(Iterator[Path]):
#     """
#     Recursively finds all *.wav under root_dir, shuffles once,
#     yields one path per iteration; on exhaustion reshuffles and continues.
#     """

#     def __init__(self, root_dir: str | Path, seed: Optional[int] = 42) -> None:
#         self.root_dir = Path(root_dir)
#         if not self.root_dir.exists():
#             raise ValueError(f"Audio root does not exist: {self.root_dir}")

#         self._files: List[Path] = list(self.root_dir.rglob("*.wav"))
#         if not self._files:
#             raise ValueError(f"No .wav files found under {self.root_dir}")

#         print(f"Found {len(self._files)} .wav files under {self.root_dir}")

#         self._rng = random.Random(seed)
#         self._idx = 0
#         self._reset()



#     def _reset(self) -> None:
#         """Reset the iterator to the beginning and reshuffle the files."""
#         self._rng.shuffle(self._files)
#         self._idx = 0

#     def __len__(self) -> int:
#         return len(self._files)

#     def __iter__(self) -> "ShuffledWavIterator":
#         return self

#     def __next__(self) -> Path:

#         if self._idx >= len(self._files):
#             self._reset()

#         p = self._files[self._idx]
#         self._idx += 1
#         return p


from __future__ import annotations
from pathlib import Path
import random
import torch
import torchaudio
from torchaudio.functional import resample

from typing import Optional, Iterator, List

from avspeech.utils.constants import SAMPLE_RATE


class ShuffledWavIterator(Iterator[torch.Tensor]):
    """
    Recursively finds all *.wav under root_dir, preloads them as tensors,
    shuffles and yields one tensor per iteration; on exhaustion reshuffles and continues.
    """

    def __init__(self, root_dir: str | Path, target_sr: int = SAMPLE_RATE, seed: Optional[int] = None) -> None:
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        
        if not self.root_dir.exists():
            raise ValueError(f"Audio root does not exist: {self.root_dir}")

        file_paths: List[Path] = list(self.root_dir.rglob("*.wav"))
        if not file_paths:
            raise ValueError(f"No .wav files found under {self.root_dir}")

        print(f"Found {len(file_paths)} .wav files under {self.root_dir}")
        
        # Load all audio files into memory
        self._audio_tensors = self._load_audio_files(file_paths)
        
        self._rng = random.Random(seed)
        self._idx = 0
        self._reset()

    def _load_audio_files(self, file_paths: List[Path]) -> List[torch.Tensor]:
        """Load all audio files from disk and convert to tensors."""
        print("Preloading audio files into memory...")
        
        audio_tensors = []
        for file_path in file_paths:
            try:
                # Use torchaudio for faster loading and direct tensor output
                waveform, orig_sr = torchaudio.load(file_path, channels_first=True)
                
                # Convert to mono if stereo (take first channel)
                if waveform.shape[0] > 1:
                    waveform = waveform[0:1, :]  # Keep first channel: [1, samples]
                
                # Resample if needed (torchaudio's sinc resampler is faster than librosa)
                if orig_sr != self.target_sr:
                    waveform = resample(waveform, orig_sr, self.target_sr)
                
                # Convert to 1D tensor (remove channel dimension)
                tensor = waveform.squeeze(0)  # [1, samples] -> [samples]
                audio_tensors.append(tensor)
                
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                continue
        
        print(f"Successfully preloaded {len(audio_tensors)} audio files")
        return audio_tensors




    def _reset(self) -> None:
        """Reset the iterator to the beginning and reshuffle the tensors."""
        self._rng.shuffle(self._audio_tensors)
        self._idx = 0

    def __len__(self) -> int:
        return len(self._audio_tensors)

    def __iter__(self) -> "ShuffledWavIterator":
        return self

    def __next__(self) -> torch.Tensor:
        if self._idx >= len(self._audio_tensors):
            self._reset()

        tensor = self._audio_tensors[self._idx]
        self._idx += 1
        return tensor