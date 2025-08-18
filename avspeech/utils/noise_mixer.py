# import ffmpeg
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from avspeech.preprocessing.audio_library_iterator import ShuffledWavIterator
from avspeech.utils.constants import SAMPLE_RATE
from avspeech.utils.structs import SampleT


class NoiseType(Enum):
    NOISE = "noise"
    MUSIC = "music"
    SPEECH = "speech"


class NoiseMixer:
    def __init__(
        self,
        audio_sources: Dict[NoiseType, ShuffledWavIterator],
        set_type: SampleT = SampleT.S1_NOISE,
    ):
        self.set_type = set_type
        self.sample_rate = SAMPLE_RATE
        self._noise_files: Dict[NoiseType, ShuffledWavIterator] = audio_sources

    @classmethod
    def from_audio_dirs(
        cls,
        noise_root: Optional[Path] = None,
        speech_root: Optional[Path] = None,
        set_type: SampleT = SampleT.S2_CLEAN,
        seed: int = 42,
    ) -> "NoiseMixer":
        """
        Convenience constructor:
        build NoiseMixer by creating ShuffledWavIterator(s) from one or both dirs.
        """
        if set_type in [SampleT.S1_NOISE, SampleT.S2_NOISE]:
            if noise_root is None:
                raise ValueError(
                    "noise_root must be provided for set_type S1_NOISE or S2_NOISE"
                )

        if set_type in [SampleT.S2_CLEAN, SampleT.S2_NOISE]:
            if speech_root is None:
                raise ValueError(
                    "speech_root must be provided for set_type S2_CLEAN or S2_NOISE"
                )

        iterators = {}
        if noise_root is not None:
            iterators[NoiseType.NOISE] = ShuffledWavIterator(noise_root, seed=seed)
        if speech_root is not None:
            iterators[NoiseType.SPEECH] = ShuffledWavIterator(speech_root, seed=seed)

        if not iterators:
            raise ValueError(
                "At least one of noise_root or speech_root must be provided."
            )

        return cls(iterators, set_type=set_type)

    def mix_1s_noise(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """Paper: AVSj + 0.3 * noise"""
        duration = len(clean_audio) / self.sample_rate
        noise = self.get_rand_noise(NoiseType.NOISE, duration)
        return mix([(clean_audio, 1.0), (noise, 0.3)], normalize=["sources"])

    def mix_2s_clean(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """Paper: AVSj + AVSk (equal amplitude speech)"""
        duration = len(clean_audio) / self.sample_rate
        other_speech = self.get_rand_noise(NoiseType.SPEECH, duration)
        return mix([(clean_audio, 1.0), (other_speech, 1.0)], normalize=["master"])

    def mix_paper_2s_noise(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """Paper: AVSj + AVSk + 0.3 * noise"""
        duration = len(clean_audio) / self.sample_rate
        noise = self.get_rand_noise(NoiseType.NOISE, duration)

        other_speech = self.get_rand_noise(NoiseType.SPEECH, duration)
        return mix(
            [(clean_audio, 1.0), (other_speech, 1.0), (noise, 0.3)],
            normalize=["master"],
        )

    def mix_with_selected_set_type(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """Mix clean audio with noise based on the specified set type."""
        if self.set_type == SampleT.S1_NOISE:
            return self.mix_1s_noise(clean_audio)
        elif self.set_type == SampleT.S2_CLEAN:
            return self.mix_2s_clean(clean_audio)
        elif self.set_type == SampleT.S2_NOISE:
            return self.mix_paper_2s_noise(clean_audio)
        else:
            raise ValueError(f"Unsupported set type: {self.set_type}")

    def get_rand_noise(self, type: NoiseType, duration: float) -> torch.Tensor:
        """Get random noise for specified duration in seconds from preloaded wav files.
        Concatanating random audio segments until the target duration is reached.
        """
        target_samples = int(duration * self.sample_rate)
        output_noise_track = torch.tensor([], dtype=torch.float32)

        sounds_iter = self._noise_files[type]

        while len(output_noise_track) < target_samples:
            # Get next preloaded audio tensor (already processed and ready to use)
            random_audio_segment = next(sounds_iter).clone()

            # Concatenate with existing noise (pure tensor operations)
            output_noise_track = torch.cat([output_noise_track, random_audio_segment])

        # Truncate to exact duration
        return output_noise_track[:target_samples]

    # def normalized_mix(self, sources_config : List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    #     """Mix various sources with specified amplitudes."""

    #     def verify_length(sound, length):
    #         if len(sound) == length:
    #             return sound

    #         print(f"Warning: Sound length mismatch! Expected {length}, got {len(sound)}")
    #         if len(sound) > length:
    #             return sound[:length]
    #         else:
    #             repeats = int(np.ceil(length / len(sound)))  # Repeat sound to match length
    #             return sound.repeat(repeats)[:length]

    #     #unzipping sources to 2 lists of 'audio_source' and 'amplitude'
    #     sources, amps = zip(*sources_config)
    #     #normalizing all sources
    #     sources = [source / source.abs().max() for source in sources]
    #     # start with a zero vector of the same length as the first source

    #     audio_length = len(sources[0])
    #     mix_result = torch.zeros(audio_length, dtype=torch.float32)
    #     # Ensure all sources matches audio length (They should come matched already, but just in case)]
    #     for sound, amp in zip(sources, amps):
    #         sound = verify_length(sound, audio_length)
    #         scaled_sound = sound * amp
    #         mix_result += scaled_sound

    #     # Clamp to prevent clipping
    #     return torch.clamp(mix_result, -1.0, 1.0).float()


def mix(
    sources_config: List[Tuple[torch.Tensor, float]],
    *,
    normalize: Optional[Sequence[str]] = None,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Mix multiple audio tensors with given amplitudes.

    Args:
        sources_config: List of (audio_tensor, amplitude) pairs.
            - audio_tensor: 1D [T] or 2D [T,2] tensor. If stereo, left channel is used.
            - amplitude: float multiplier.
        normalize:
            None             -> no normalization
            ["sources"]      -> normalize each source individually
            ["mix"]          -> normalize only the final mix
            ["sources","mix"]-> normalize sources first, then normalize final mix again
        clamp: If True, clamp final mix to [-1, 1].

    Returns:
        torch.Tensor: 1D float32 mono mix.
    """
    if not sources_config:
        raise ValueError(
            "sources_config must contain at least one (tensor, amplitude) pair."
        )

    normalize = set(normalize or [])

    def _to_mono(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x
        elif x.ndim == 2 and x.shape[1] >= 1:
            return x[:, 0]  # left channel
        else:
            raise ValueError(f"Unexpected audio shape {tuple(x.shape)}")

    def _verify_length(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.shape[0] == length:
            return x
        if x.shape[0] > length:
            return x[:length]
        repeats = int(np.ceil(length / x.shape[0]))
        return x.repeat(repeats)[:length]

    def _peak_normalize(x: torch.Tensor) -> torch.Tensor:
        peak = x.abs().max()
        if peak == 0 or torch.isnan(peak):
            return x
        return x / peak

    # Unpack
    sources, amps = zip(*sources_config)
    sources = [_to_mono(s) for s in sources]

    target_len = sources[0].shape[0]
    device = sources[0].device

    master = torch.zeros(target_len, dtype=torch.float32, device=device)

    for src, amp in zip(sources, amps):
        src = src.to(device)

        if "sources" in normalize:
            src = _peak_normalize(src)

        src = _verify_length(src, target_len)
        master += src.to(torch.float32) * float(amp)

    if "master" in normalize:
        master = _peak_normalize(master)

    if clamp:
        master = torch.clamp(master, -1.0, 1.0)

    return master.float()
