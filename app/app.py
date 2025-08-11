"""
Gradio app integration using the clean flow.
"""
from pathlib import Path

import gradio as gr
from typing import Optional

# Import the clean flow functions
from app_flow import (
    analyze_video_segment,
    process_with_face_selection,
    get_status_message,
    auto_proceed_if_single_face,
    AnalysisResult
)

# Import other needed components
from avspeech.preprocessing.pipeline import init_models
from avspeech.utils.audio import process_audio_for_inference


class AppState:
    """Simple state management for the app."""

    def __init__(self):
        self.face_embedder = init_models()
        self.current_analysis: Optional[AnalysisResult] = None
        self.face_embeddings = None
        self.audio_chunks = None


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
                    video_file,
                    start_time,
                    end_time,
                    app_state.face_embedder
            )
        except Exception as e:
            return None, f"Error in 'analyze_video_segment' (app_flow.py): {str(e)}", gr.update(visible=False)


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
            app_state.face_embeddings, _ = process_with_face_selection(
                    analysis,
                    app_state.face_embedder,
                    selected_face_id=None
            )
        else:
            # Multiple faces or no faces - need user action
            #TODO - If no face was detected, show a message, do not proceed
            show_reduce = len(analysis.best_frame_boxes) > 0
            if show_reduce:
                status += "\n👆 Click on a face to select it"

        return (
                analysis.preview_image,
                status,
                gr.update(visible=show_reduce)
        )

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
        if (box['x'] <= x <= box['x'] + box['width'] and
                box['y'] <= y <= box['y'] + box['height']):
            selected_face_id = box['face_id']
            break

    if selected_face_id is None:
        return "No face at that location. Try clicking on a face box."

    # Process with selected face
    app_state.face_embeddings, hint = process_with_face_selection(
            app_state.current_analysis,
            app_state.face_embedder,
            selected_face_id
    )

    return f"✅ Selected Face {selected_face_id + 1}. Ready for noise reduction."


def handle_reduce_noise():
    """
    Handle the Reduce Noise button click.
    """
    if not app_state.current_analysis:
        return None, "Please analyze segment first"

    if not app_state.face_embeddings:
        return None, "Please select a face first (or re-analyze)"

    try:
        # Process audio (if not already done)
        if not app_state.audio_chunks:
            # TODO - Keep the original chunks ('_') for later showing some metrics
            app_state.audio_chunks, _ = process_audio_for_inference(
                    Path(app_state.current_analysis.segment_path)
            )

        # TODO: Run actual inference
        # enhanced_audio = run_inference(
        #     app_state.audio_chunks,
        #     app_state.face_embeddings
        # )

        # For now, return placeholder
        return (
                app_state.current_analysis.segment_path,  # Return original for now
                f"✅ Ready for inference!\n"
                f"Audio chunks: {len(app_state.audio_chunks)}\n"
                f"Face embeddings: {len(app_state.face_embeddings)} chunks"
        )

    except Exception as e:
        return None, f"Error: {str(e)}"


# Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AV-Speech Noise Reduction")

    with gr.Row():
        # Left column: Input
        with gr.Column():
            video_input = gr.Video(label="Upload Video")

            with gr.Row():
                start_time = gr.Slider(0, 60, 0, label="Start (s)")
                end_time = gr.Slider(0, 60, 10, label="End (s)")

            analyze_btn = gr.Button("🔍 Analyze", variant="secondary")
            status = gr.Textbox(label="Status", interactive=False)
            reduce_btn = gr.Button("🔊 Reduce Noise", variant="primary", visible=False)

        # Right column: Preview and Output
        with gr.Column():
            preview = gr.Image(label="Face Detection Preview", interactive=True)
            output_video = gr.Video(label="Enhanced Output")
            output_status = gr.Textbox(label="Processing Status", interactive=False)

    # Event handlers
    analyze_btn.click(
            handle_analyze,
            inputs=[video_input, start_time, end_time],
            outputs=[preview, status, reduce_btn]
    )

    # Click on preview to select face
    preview.select(
            handle_face_click,
            outputs=[status]
    )

    reduce_btn.click(
            handle_reduce_noise,
            outputs=[output_video, output_status]
    )

if __name__ == "__main__":
    demo.launch()