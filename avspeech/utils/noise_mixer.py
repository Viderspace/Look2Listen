import torch
import soundfile as sf
import librosa
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Tuple
# import ffmpeg

from enum import Enum

from avspeech.utils.structs import SampleT


class NoiseType(Enum):
    NOISE = "noise"
    MUSIC = "music"
    SPEECH = "speech"



class NoiseMixer:

    def __init__(self, noise_root: Path, set_type : SampleT, sample_rate: int):
        self.noise_root = noise_root
        self.set_type = set_type
        self.sample_rate = sample_rate
        self._noise_files = self._load_noise_files()

        print(f"NoiseMixer initialized with noise root: {self.noise_root}, set type: {self.set_type}, sample rate: {self.sample_rate}")



    def _load_noise_files(self) -> Dict[NoiseType, List[Path]]:
        """Load noises files for sub directories in noise_root based on specified noise types. """
        # sub_directories : list[Path] = [] # making sub_directories path for each noise type
        noise_files : Dict[NoiseType, List[Path]] = {}

        misc_noises_path = self.noise_root / NoiseType.NOISE.value
        noise_files[NoiseType.NOISE] = list(misc_noises_path.rglob("*.wav"))

        speech_noises_path = self.noise_root / NoiseType.SPEECH.value
        noise_files[NoiseType.SPEECH] = list(speech_noises_path.rglob("*.wav"))

        music_noises_path = self.noise_root / NoiseType.MUSIC.value
        noise_files[NoiseType.MUSIC] = list(music_noises_path.rglob("*.wav"))

        print(f"NoiseMixer: Loaded {len(noise_files[NoiseType.NOISE])} noise samples, {len(noise_files[NoiseType.SPEECH])} speech samples, and {len(noise_files[NoiseType.MUSIC])} samples.")
        return noise_files

    def mix_1s_noise(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """Paper: AVSj + 0.3 * noise"""
        duration = len(clean_audio) / self.sample_rate
        noise = self.get_rand_noise(NoiseType.NOISE ,duration)
        return self.normalized_mix([(clean_audio, 1.0), (noise, 0.3)])

    def mix_2s_clean(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """Paper: AVSj + AVSk (equal amplitude speech)"""
        duration = len(clean_audio) / self.sample_rate
        other_speech = self.get_rand_noise(NoiseType.SPEECH, duration)
        return self.normalized_mix([(clean_audio, 1.0), (other_speech, 1.0)])


    def mix_paper_2s_noise(self, clean_audio: torch.Tensor) -> torch.Tensor:
        """Paper: AVSj + AVSk + 0.3 * noise"""
        duration = len(clean_audio) / self.sample_rate
        noise = self.get_rand_noise(NoiseType.NOISE, duration)

        other_speech = self.get_rand_noise(NoiseType.SPEECH, duration)
        return self.normalized_mix([(clean_audio, 1.0), (other_speech, 1.0), (noise, 0.3)])

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

    def get_rand_noise(self, type : NoiseType, duration: float) -> torch.Tensor:
        """Get random noise for specified duration in seconds."""
        target_samples = int(duration * self.sample_rate)
        noise_data = np.array([])

        while len(noise_data) < target_samples:
            # Select random noise file
            noise_file = random.choice(self._noise_files[type])

            try:
                data, sr = sf.read(noise_file, dtype='float32')

                # Convert to mono (take left channel if stereo)
                if data.ndim > 1:
                    data = data[:, 0]

                # Resample if needed
                if sr != self.sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)

                if type in [NoiseType.SPEECH, NoiseType.MUSIC]:
                    # Randomizing long vocal/music samples by taking a random segment from it
                    start = random.randint(0, len(data))
                    data = data[start:]



                # Concatenate with existing noise
                noise_data = np.concatenate([noise_data, data])


            except Exception as e:
                print(f"Error loading {noise_file.name}: {e}")
                continue

        # Truncate to exact duration
        noise_data = noise_data[:target_samples]
        return torch.from_numpy(noise_data).float()

    def normalized_mix(self, sources_config : List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Mix various sources with specified amplitudes."""

        def verify_length(sound, length):
            if len(sound) == length:
                return sound

            print(f"Warning: Sound length mismatch! Expected {length}, got {len(sound)}")
            if len(sound) > length:
                return sound[:length]
            else:
                repeats = int(np.ceil(length / len(sound)))  # Repeat sound to match length
                return sound.repeat(repeats)[:length]

        #unzipping sources to 2 lists of 'audio_source' and 'amplitude'
        sources, amps = zip(*sources_config)
        #normalizing all sources
        sources = [source / source.abs().max() for source in sources]
        # start with a zero vector of the same length as the first source

        audio_length = len(sources[0])
        mix_result = torch.zeros(audio_length, dtype=torch.float32)
        # Ensure all sources matches audio length (They should come matched already, but just in case)]
        for sound, amp in zip(sources, amps):
            sound = verify_length(sound, audio_length)
            scaled_sound = sound * amp
            mix_result += scaled_sound

        # Clamp to prevent clipping
        return torch.clamp(mix_result, -1.0, 1.0).float()



    # def render_noised_mp4_clip(self, clean_audio: torch.Tensor, mp4_path: Path, output_dir_path: Path,
    #                            amp: float = 0.3):
    #     """Generate random noise and save as MP4 with noised audio."""
    #
    #     print(f"{clean_audio.shape = }")
    #
    #     # Get random noise
    #     clip_duration = len(clean_audio) / self.sample_rate
    #     # Mix clean + noise
    #     mixed_audio = self.mix_with_selected_set_type(clean_audio)
    #
    #     # Save temporary mixed audio
    #     temp_audio_path = output_dir_path / f"{mp4_path.stem}_temp_noised.wav"
    #     sf.write(temp_audio_path, mixed_audio.numpy(), self.sample_rate)
    #
    #     # Prepare output video path
    #     output_file = output_dir_path / f"{mp4_path.stem}_noised.mp4"
    #
    #     # Use ffmpeg to replace audio
    #     try:
    #         video_input = ffmpeg.input(str(mp4_path))
    #         audio_input = ffmpeg.input(str(temp_audio_path))
    #
    #         # Method 1: Using separate map calls (recommended)
    #         stream = ffmpeg.output(
    #                 video_input['v'],  # Video stream from first input
    #                 audio_input['a'],  # Audio stream from second input
    #                 str(output_file),
    #                 vcodec='copy',  # Copy video without re-encoding
    #                 acodec='aac',  # Re-encode audio as AAC
    #                 audio_bitrate='128k'  # Optional: set audio bitrate
    #         )
    #
    #
    #         stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
    #
    #     except ffmpeg.Error as e:
    #         print("FFmpeg error:", e.stderr.decode())
    #         print("FFmpeg stdout:", e.stdout.decode())
    #         raise
    #     except Exception as e:
    #         print(f"Unexpected error: {e}")
    #         raise
    #
    #     # Clean up temporary file
    #     if temp_audio_path.exists():
    #         temp_audio_path.unlink()
    #
    #     return output_file