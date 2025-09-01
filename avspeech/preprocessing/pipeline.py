"""
Preprocessing Pipeline Orchestrator
Coordinates audio and video processing for training and inference
"""

from pathlib import Path
from typing import List, Tuple
import torch

from avspeech import FaceEmbedder
from avspeech.preprocessing.clips_loader import ClipData
from avspeech.preprocessing.export import save_processed_clip

from avspeech.utils.noise_mixer import NoiseMixer
# from avspeech.utils.face_embedder import FaceEmbedder
from avspeech.utils.FaceManager import FaceManager
from avspeech.utils.audio import process_audio_for_training, process_audio_for_inference
from avspeech.utils.video import process_video_for_training, process_video_for_inference
from avspeech.utils.structs import AudioChunk, Sample


# ─────────────────────── Configuration ─────────────────────────

# Expected tensor shapes for validation
AUDIO_CHUNK_SHAPE = (257, 298, 2)  # STFT features
# FACE_EMBEDDING_SHAPE = (75, 512)  # 75 frames, 512-D embeddings
FACE_EMBEDDING_SHAPE = (75, 1792)  # 75 frames, 1792-D embeddings *Newer approach*

EXPECTED_DTYPE = torch.float32


# ─────────────────────── Initialization ─────────────────────────

def init_models() -> FaceEmbedder:
    """Initialize the face embedder model."""
    return FaceEmbedder()


# ─────────────────────── Core Processing Functions ─────────────────────────

def process_for_training(
        clip_data: ClipData,
        face_manager: FaceManager,
        noise_mixer: NoiseMixer,
        output_dir: Path
) -> int:
    """
    Process a video clip for training dataset creation.

    Args:
        clip_data: Video metadata and path
        face_manager: Face detection/embedding model
        noise_mixer: Audio noise augmentation
        output_dir: Where to save processed chunks

    Returns:
        Number of chunks successfully saved
    """
    video_path = clip_data.video_path_obj

    # Check video exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    try:

    # Process audio and video pipelines
        audio_chunks : List[AudioChunk] = process_audio_for_training(video_path, noise_mixer)

    except Exception as e:
        print(f"process_for_training() - Error processing audio {clip_data.unique_clip_id}: {e}")
        return 0
    
    try:
        video_chunks : List[torch.Tensor] = process_video_for_training(
                mp4_path= clip_data.video_path_obj,
                face_manager=face_manager,
                face_hint_pos=(clip_data.clip_metadata.get("x", 0.5), clip_data.clip_metadata.get("y", 0.5)))

    except Exception as e:
        print(f"process_for_training() - Error processing video {clip_data.unique_clip_id}: {e}")
        return 0

    # Validate alignment
    if not _validate_chunks(audio_chunks, video_chunks):
        raise ValueError(f"Chunk validation failed for {clip_data.unique_clip_id}")

    print(f"faced embs output chunks are: {video_chunks[0].shape}, dtype: {video_chunks[0].dtype}, max: {video_chunks[0].max()}, min: {video_chunks[0].min()}")

    # Packaging embeddings for saving
    samples = [Sample(face=v, mix=a.mixed, clean=a.clean) for v, a in zip(video_chunks, audio_chunks)]
    return save_processed_clip(samples, clip_data.unique_clip_id, output_dir)



def process_for_inference(
        mp4_path: Path,
        face_manager: FaceManager,
        user_hint: [Tuple[float, float]] = (0.5, 0.5)  # Default hint. center is enough if only 1 face
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:


    # Check video exists
    if not mp4_path.exists():
        raise FileNotFoundError(f"Video not found: {mp4_path}")

    # Process pipelines
    audio_embeddings = process_audio_for_inference(mp4_path)
    face_embeddings = process_video_for_inference(mp4_path, face_manager, user_hint)

    # Validate alignment
    if len(audio_embeddings) != len(face_embeddings):
        raise ValueError(
                f"Chunk mismatch: {len(audio_embeddings)} audio != {len(face_embeddings)} video"
        )

    # Return inference-ready data
    return [
            (idx, face, audio)
            for idx, (face, audio)
            in enumerate(zip(face_embeddings, audio_embeddings))
    ]


# ─────────────────────── Validation ─────────────────────────

def _validate_chunks(
        audio_chunks: List[AudioChunk],
        video_chunks: List[torch.Tensor]
) -> bool:
    """
    Validate chunk alignment and shapes.

    Returns:
        True if valid, False otherwise
    """
    # Check we have chunks
    if not audio_chunks or not video_chunks:
        print(f"No chunks to validate: audio={len(audio_chunks)}, video={len(video_chunks)}")
        return False

    # Check counts match
    if len(audio_chunks) != len(video_chunks):
        print(f"Chunk count mismatch: audio={len(audio_chunks)}, video={len(video_chunks)}")
        return False

    # Check shapes
    for audio_chunk, video_chunk in zip(audio_chunks, video_chunks):
        # Check audio shapes
        if audio_chunk.clean.shape != AUDIO_CHUNK_SHAPE:
            print(f"Invalid audio clean shape: {audio_chunk.clean.shape}")
            return False
        if audio_chunk.mixed.shape != AUDIO_CHUNK_SHAPE:
            print(f"Invalid audio mixed shape: {audio_chunk.mixed.shape}")
            return False

        # Check video shape
        if video_chunk.shape != FACE_EMBEDDING_SHAPE:
            print(f"Invalid video embedding shape: {video_chunk.shape}")
            return False

        # Check dtypes
        if audio_chunk.clean.dtype != EXPECTED_DTYPE:
            return False
        if video_chunk.dtype != EXPECTED_DTYPE:
            return False

    return True




# ─────────────────────── Convenience Wrappers ─────────────────────────

def process_single_clip(
        face_manager: FaceManager,
        clip_data: ClipData,
        noise_mixer: NoiseMixer,
        output_dir: Path
) -> int:
    """
    Legacy wrapper for training processing.
    Maintained for backward compatibility.
    """
    try:
        return process_for_training(
                clip_data=clip_data,
                face_manager=face_manager,
                noise_mixer=noise_mixer,
                output_dir=output_dir
        )
    except Exception as e:
        print(f"process_single_clip() - Error processing {clip_data.unique_clip_id}: {e}")
        return 0


def pre_process_for_inference(
        face_manager: FaceManager,
        clip_data: ClipData
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Legacy wrapper for inference processing.
    Maintained for backward compatibility.
    """
    try:
        return process_for_inference(
                mp4_path=clip_data.video_path_obj,
                face_manager=face_manager,
                user_hint=(clip_data.clip_metadata.get("x", 0.5), clip_data.clip_metadata.get("y", 0.5))
        )
    except Exception as e:
        print(f"pre_process_for_inference() - Error processing {clip_data.unique_clip_id}: {e}")
        return []

