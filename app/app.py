"""
Gradio app integration using the clean flow.
"""

from pathlib import Path
from typing import Optional

import gradio as gr

# Import the clean flow functions
from app_flow import (
    AnalysisResult,
    analyze_video_segment,
    auto_proceed_if_single_face,
    get_status_message,
    process_with_face_selection,
)

# Import other needed components
from avspeech.preprocessing.pipeline import init_models


class AppState:
    """Simple state management for the app."""

    def __init__(self):
        self.face_embedder = init_models()
        self.current_analysis: Optional[AnalysisResult] = None
        self.face_embeddings = None
        self.audio_chunks = None
        self.face_hint_coords: Optional[tuple] = None
        # Use bundled checkpoint
        self.checkpoint_path = str(
            Path(__file__).parent / "models" / "checkpoint_epoch_26.pt"
        )


# Initialize app state
app_state = AppState()


def handle_analyze(video_file, start_time, end_time):
    """
    Handle the Analyze button click.
    """
    if not video_file:
        return None, "Please upload a video", gr.update(visible=False)

    try:
        try:
            # Step 1: Analyze
            analysis = analyze_video_segment(
                video_file, start_time, end_time, app_state.face_embedder
            )
        except Exception as e:
            return (
                None,
                f"Error in 'analyze_video_segment' (app_flow.py): {str(e)}",
                gr.update(visible=False),
            )

        # Store in state
        app_state.current_analysis = analysis

        # Get status
        status = get_status_message(analysis)

        # Check if we should show face selection or auto-proceed
        if auto_proceed_if_single_face(analysis):
            # Auto-proceed with single face
            status += "\n➡️ Auto-proceeding with single face"
            show_reduce = True

            # Auto-generate embeddings
            app_state.face_embeddings, hint_coords = process_with_face_selection(
                analysis, app_state.face_embedder, selected_face_id=None
            )
            # Store hint coordinates for ClipProcessor
            app_state.face_hint_coords = hint_coords
        else:
            # Multiple faces or no faces - need user action
            # TODO - If no face was detected, show a message, do not proceed
            show_reduce = len(analysis.best_frame_boxes) > 0
            if show_reduce:
                status += "\n👆 Click on a face to select it"

        return (analysis.preview_image, status, gr.update(visible=show_reduce))

    except Exception as e:
        return None, f"Error: {str(e)}", gr.update(visible=False)


def handle_face_click(evt: gr.SelectData):
    """
    Handle click on the preview image to select a face.
    """
    if not app_state.current_analysis:
        return "Please analyze first"

    # Get click coordinates
    x, y = evt.index

    # Find which face was clicked
    selected_face_id = None
    for box in app_state.current_analysis.best_frame_boxes:
        if (
            box["x"] <= x <= box["x"] + box["width"]
            and box["y"] <= y <= box["y"] + box["height"]
        ):
            selected_face_id = box["face_id"]
            break

    if selected_face_id is None:
        return "No face at that location. Try clicking on a face box."

    # Process with selected face to get hint coordinates
    app_state.face_embeddings, hint_coords = process_with_face_selection(
        app_state.current_analysis, app_state.face_embedder, selected_face_id
    )

    # Store hint coordinates for ClipProcessor
    app_state.face_hint_coords = hint_coords

    return f"✅ Selected Face {selected_face_id + 1}. Ready for noise reduction."


def handle_reduce_noise():
    """
    Handle the Reduce Noise button click.
    """
    if not app_state.current_analysis:
        return None, "Please analyze segment first"

    if not app_state.face_hint_coords:
        return None, "Please select a face first (or re-analyze)"

    if not Path(app_state.checkpoint_path).exists():
        return None, "Error: Bundled model checkpoint not found"

    try:
        from avspeech.utils.ClipProcessor import ClipProcessor

        # Create ClipProcessor from the video segment
        clip_processor = ClipProcessor(Path(app_state.current_analysis.segment_path))

        if not clip_processor.is_video_loaded():
            return None, "Error: Failed to load video segment"

        # Set face hint from UI selection
        clip_processor.set_face_hint(app_state.face_hint_coords)

        if not clip_processor.is_video_ready():
            return None, "Error: Video processing failed. Cannot proceed with inference"

        # Run inference using the complete pipeline
        enhanced_audio, original_audio = clip_processor.apply_inference(
            Path(app_state.checkpoint_path)
        )

        # Get original video info for trimming
        original_video_path = Path(app_state.current_analysis.segment_path)
        video_info = get_video_info(original_video_path)

        # Trim audio to match original video duration (remove padding)
        enhanced_audio_trimmed = trim_audio_to_video_length(
            enhanced_audio, video_info["duration"]
        )
        original_audio_trimmed = trim_audio_to_video_length(
            original_audio, video_info["duration"]
        )
        print(f"original peak : {original_audio_trimmed.abs().max()}, "
              f"enhanced peak: {enhanced_audio_trimmed.abs().max()}")

        from avspeech.utils.audio import normalize_audio
        # Normalize audio to prevent clipping
        enhanced_audio_trimmed = normalize_audio(enhanced_audio_trimmed, )
        original_audio_trimmed = normalize_audio(original_audio_trimmed)
        print(f"original peak : {original_audio_trimmed.abs().max()}, "
              f"enhanced peak: {enhanced_audio_trimmed.abs().max()}")

        # Create output directory
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        # Generate output filenames
        video_name = Path(original_video_path).stem
        enhanced_video_path = temp_dir / f"{video_name}_enhanced.mp4"
        original_video_path_copy = temp_dir / f"{video_name}_original.mp4"

        # Save enhanced video (video + enhanced audio)
        save_audio_video_combined(
            original_video_path, enhanced_audio_trimmed, enhanced_video_path
        )

        # Save original video (video + original audio) for comparison
        save_audio_video_combined(
            original_video_path, original_audio_trimmed, original_video_path_copy
        )



        return (
            str(enhanced_video_path),  # Return enhanced video instead of just audio
            f"✅ Audio enhancement completed!\n"
            f"Model: checkpoint_epoch_32 (bundled)\n"
            f"Original duration: {video_info['duration']:.2f}s\n"
            f"Enhanced duration: {enhanced_audio_trimmed.shape[-1] / 16000:.2f}s\n"
            f"Trimmed padding and combined with video\n"
            f"Enhanced video ready for download!",
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"Error during inference: {str(e)}"


def get_model_info():
    """Get information about the bundled model."""
    checkpoint_path = Path(app_state.checkpoint_path)
    if checkpoint_path.exists():
        return f"✅ Model loaded: {checkpoint_path.name} (bundled)"
    else:
        return "❌ Bundled model not found"


def get_video_info(video_path: Path) -> dict:
    """Get basic video information for trimming purposes."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return {"fps": fps, "frame_count": frame_count, "duration": duration}


def trim_audio_to_video_length(
    audio_waveform, video_duration: float, sample_rate: int = 16000
):
    """Trim audio waveform to match original video duration."""
    target_length = int(video_duration * sample_rate)
    if audio_waveform.size(-1) > target_length:
        return audio_waveform[..., :target_length]
    return audio_waveform


def save_audio_video_combined(
    video_path: Path, audio_waveform, output_path: Path, sample_rate: int = 16000
):
    """Combine video frames with new audio and save as MP4."""
    import ffmpeg
    import torchaudio

    # Save audio as temporary WAV file
    temp_audio_path = output_path.with_suffix(".temp.wav")

    # Ensure audio has correct shape for torchaudio.save
    if len(audio_waveform.shape) == 1:
        audio_waveform = audio_waveform.unsqueeze(0)

    torchaudio.save(str(temp_audio_path), audio_waveform, sample_rate)

    try:
        # Create ffmpeg inputs
        video_input = ffmpeg.input(str(video_path))
        audio_input = ffmpeg.input(str(temp_audio_path))

        # Combine video and audio
        stream = ffmpeg.output(
            video_input["v"],  # Video stream from first input
            audio_input["a"],  # Audio stream from second input
            str(output_path),
            vcodec="copy",  # Copy video without re-encoding
            acodec="aac",  # Re-encode audio as AAC
            audio_bitrate="320k",
            shortest=None,  # Match shortest stream
        )

        stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
        return True

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode() if e.stderr else "Unknown")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
    finally:
        # Clean up temporary audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()


# Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AV-Speech Noise Reduction")
    gr.Markdown(
        "*Upload a video, select a time segment, and let AI enhance the speech by reducing background noise.*"
    )

    with gr.Row():
        # Left column: Input
        with gr.Column():
            video_input = gr.Video(label="Upload Video")

            with gr.Row():
                start_time = gr.Slider(0, 60, 0, label="Start (s)")
                end_time = gr.Slider(0, 60, 10, label="End (s)")

            model_info = gr.Textbox(
                label="Model Status", value=get_model_info(), interactive=False
            )

            analyze_btn = gr.Button("🔍 Analyze Video Segment", variant="secondary")
            status = gr.Textbox(label="Analysis Status", interactive=False)
            reduce_btn = gr.Button("🔊 Enhance Audio", variant="primary", visible=False)

        # Right column: Preview and Output
        with gr.Column():
            preview = gr.Image(label="Face Detection Preview", interactive=True)
            output_video = gr.Video(label="Enhanced Video Output")
            output_status = gr.Textbox(label="Processing Status", interactive=False)

            with gr.Row():
                gr.Markdown(
                    "*Enhanced video includes original video frames with AI-processed audio*"
                )

    # Event handlers
    # No checkpoint upload needed - using bundled model

    analyze_btn.click(
        handle_analyze,
        inputs=[video_input, start_time, end_time],
        outputs=[preview, status, reduce_btn],
    )

    # Click on preview to select face
    preview.select(handle_face_click, outputs=[status])

    reduce_btn.click(handle_reduce_noise, outputs=[output_video, output_status])

if __name__ == "__main__":
    demo.launch()


"""
Epoch 24: loss=0.0905
Running full validation...
Validating 1s_noise with 71 batches...

  Val 1s_noise: loss=0.0589
Validating 2s_clean with 75 batches...
  Val 2s_clean: loss=0.1005
Validating 2s_noise with 73 batches...
  Val 2s_noise: loss=0.1021
  """