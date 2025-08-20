from pathlib import Path
from typing import List, Optional, Tuple

import torch

import avspeech.utils.fourier_transform as fourier
from avspeech import FaceEmbedder
from avspeech.model.av_model import AudioVisualModel
from avspeech.utils.audio import (
    add_noise_to_audio,
    extract_audio,
    process_audio_for_inference,
    process_audio_with_noise,
)
from avspeech.utils.face_render_tests import save_debug_collage
from avspeech.utils.structs import SampleT
from avspeech.utils.video import extract_frames_for_inference


def render_noised_mp4_clip(mp4_path: Path, output_dir_path: Path):
    """Generate random noise and save as MP4 with noised audio."""
    global ffmpeg
    audio = extract_audio(mp4_path, mode="inference")
    mixes = add_noise_to_audio(audio)  # Process audio with noise augmentation
    import soundfile as sf

    # Save temporary mixed audio
    temp_audio_path_1s = output_dir_path / f"{mp4_path.stem}_temp_noised_1s.wav"
    temp_audio_path_2sc = output_dir_path / f"{mp4_path.stem}_temp_noised_2sc.wav"
    temp_audio_path_2sn = output_dir_path / f"{mp4_path.stem}_temp_noised_2sn.wav"

    sf.write(temp_audio_path_1s, mixes[0].numpy(), 16000)
    sf.write(temp_audio_path_2sc, mixes[1].numpy(), 16000)
    sf.write(temp_audio_path_2sn, mixes[2].numpy(), 16000)

    # Prepare output video path
    out_1s = output_dir_path / f"{mp4_path.stem}_noised_1s.mp4"
    out_2sc = output_dir_path / f"{mp4_path.stem}_noised_2sc.mp4"
    out_2sn = output_dir_path / f"{mp4_path.stem}_noised_2sn.mp4"

    # Use ffmpeg to replace audio
    for temp_audio_path, output_file in [
        (temp_audio_path_1s, out_1s),
        (temp_audio_path_2sc, out_2sc),
        (temp_audio_path_2sn, out_2sn),
    ]:
        try:
            import ffmpeg

            video_input = ffmpeg.input(str(mp4_path))
            audio_input = ffmpeg.input(str(temp_audio_path))

            # Method 1: Using separate map calls (recommended)
            stream = ffmpeg.output(
                video_input["v"],  # Video stream from first input
                audio_input["a"],  # Audio stream from second input
                str(output_file),
                vcodec="copy",  # Copy video without re-encoding
                acodec="aac",  # Re-encode audio as AAC
                audio_bitrate="320k",  # Optional: set audio bitrate
                shortest=None,  # Match shortest stream
            )

            stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)

        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode())
            print("FFmpeg stdout:", e.stdout.decode())
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

        # Clean up temporary file
        if temp_audio_path.exists():
            temp_audio_path.unlink()

    return out_1s, out_2sc, out_2sn


class ClipProcessor:
    """
    Complete video clip represented as embeddings and metadata.
    """

    def __init__(self, video_path: Path, add_noise: Optional[SampleT] = None):
        """
        Initialize the ClipProcessor with a video file path.
        """
        self.face_crops = None
        self.device = torch.device("cpu")
        self.video_path = video_path
        self.video_frames = extract_frames_for_inference(video_path)
        self.audio_embeddings = self.process_audio(video_path, add_noise)
        self.face_embeddings = []
        self.inference_samples = []
        self.face_hint: Optional[Tuple[float, float]] = (
            None  # User-provided hint for face cropping
        )
        self.inference_results = []
        self.inference_masks = []  # not used yet, but can be useful for debugging

        assert self.is_video_loaded(), (
            f"ClipProcessor: Failed to load video {video_path.name}. Check if the file exists and is valid."
        )
        # self.render_noised_mp4_clip(video_path, video_path.parent)

    def is_video_loaded(self) -> bool:
        return (
            self.video_frames
            and len(self.video_frames) > 0
            and self.audio_embeddings
            and len(self.audio_embeddings) > 0
        )

    def is_video_ready(self) -> bool:
        """
        Check if the video is ready for inference by verifying that both audio and face embeddings are available,
        and a face hint has been set. Report any issues with the video processing.
        """
        if not self.video_frames or len(self.video_frames) == 0:
            print(
                f"ClipProcessor: No video frames extracted for {self.video_path.name}"
            )
            return False
        if not self.audio_embeddings or len(self.audio_embeddings) == 0:
            print(
                f"ClipProcessor: No audio embeddings available for {self.video_path.name}"
            )
            return False
        if not self.face_hint:
            print(f"ClipProcessor: No face hint set for {self.video_path.name}")
            return False
        if not self.face_embeddings or len(self.face_embeddings) == 0:
            print(
                f"ClipProcessor: No face embeddings computed for {self.video_path.name}"
            )
            return False
        if len(self.face_embeddings) != len(self.audio_embeddings):
            print(
                f"ClipProcessor: Chunk mismatch: {len(self.face_embeddings)} face embeddings != {len(self.audio_embeddings)} audio embeddings"
            )
            return True  # we can still run inference, eventually the encoding to mp4 will pick the shortest source

        return True

    def process_audio(
        self, video_path: Path, add_noise: Optional[SampleT] = None
    ) -> List[torch.Tensor]:
        if add_noise is None:
            return process_audio_for_inference(
                video_path
            )  # Process audio without noise augmentation

        return process_audio_with_noise(
            video_path, add_noise
        )  # Process audio with noise augmentation

    def set_face_hint(self, hint_pos: Tuple[float, float] = (0.5, 0.5)):
        """
        By receiving a hint position (usually selected by the user) the clip processor now can create face embeddings
        from the video frames.
        """

        if not self.video_frames or len(self.video_frames) == 0:
            raise ValueError(f"No video frames extracted for {self.video_path.name}")

        self.face_hint = hint_pos

        face_embedder = FaceEmbedder()
        self.face_crops = face_embedder.crop_faces(
            self.video_frames, hint_pos[0], hint_pos[1]
        )
        save_debug_collage(Path(f"./debug/collage_{self.video_path.name}_{hint_pos}"), self.face_crops)

        self.face_embeddings = face_embedder.compute_embeddings(self.face_crops)

        assert self.is_video_ready()

    def _infer_chunks(
        self,
        audio_chunk: torch.Tensor,
        face_chunk: torch.Tensor,
        model: AudioVisualModel,
    ):
        """
        Run inference on a single chunk of audio and face embeddings.
        """
        with torch.no_grad():
            audio_chunk = audio_chunk.to(self.device)
            face_chunk = face_chunk.to(self.device)

            if len(audio_chunk.shape) == 3:  # Add batch dim if missing
                audio_chunk = audio_chunk.unsqueeze(0)

            if len(face_chunk.shape) == 2:  # Add batch dim if missing
                face_chunk = face_chunk.unsqueeze(0)

            # Run model - outputs mask for target speaker
            mask = model(audio_chunk, face_chunk)

            # Apply mask to enhance target speaker
            inferred_audio_chunk = audio_chunk * mask

            assert inferred_audio_chunk.shape == audio_chunk.shape, (
                "ClipProcessor: Inference output shape mismatch"
            )

            return inferred_audio_chunk.squeeze(0), mask.squeeze(0)

    def apply_inference(self, model_weights_file_path: Path):
        """
        Apply inference model to the audio embeddings and store the results.
        """
        if not self.is_video_ready():
            raise ValueError(
                "ClipProcessor: Video is not ready for inference. Please set face hint and ensure embeddings are available."
            )

        model = AudioVisualModel().to(self.device)
        checkpoint = torch.load(
            model_weights_file_path, map_location=self.device, weights_only=True
        )  # Load trained weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        for audio_chunk, face_chunk in zip(self.audio_embeddings, self.face_embeddings):
            enhanced_audio, mask = self._infer_chunks(audio_chunk, face_chunk, model)
            self.inference_results.append(enhanced_audio)

        # Concatenate all STFTs along time dimension [257, 298*5, 2]
        full_clip_enhanced_stft = torch.cat(self.inference_results, dim=1)
        full_clip_original_stft = torch.cat(self.audio_embeddings, dim=1)

        full_enhanced = fourier.stft_to_audio(full_clip_enhanced_stft)
        full_original = fourier.stft_to_audio(full_clip_original_stft)

        # remove silent padding samples from the end to leave only the original audio length - here

        return full_enhanced, full_original
