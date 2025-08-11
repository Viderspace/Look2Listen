"""
timer_decorator.py
==================

Timer decorator for measuring and logging processing step durations.
Provides consistent timing logs across the preprocessing pipeline.
"""

import time
import functools
import logging


def timer(step_name: str = None):
    """
    Decorator that logs the runtime of pipeline steps with consistent formatting.

    Args:
        step_name: Custom name for the step (defaults to formatted function name)

    Usage:
        @timer("Audio embeddings")
        def process_audio_pipeline(mp4_path, noise_mixer):
            return create_audio_embeddings(mp4_path, noise_mixer)

        @timer()  # Uses function name
        def extract_video_frames(mp4_path):
            return extract_frames(mp4_path)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use custom name or format function name
            name = step_name or func.__name__.replace('_', ' ').title()

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            # Match existing log format
            logging.info(f"  ⏱ {name}: {duration:.2f}s")
            return result

        return wrapper

    return decorator


def timer_with_info(step_name: str = None, info_func=None):
    """
    Enhanced timer decorator that can log additional information.

    Args:
        step_name: Custom name for the step
        info_func: Function that takes the result and returns additional info string

    Usage:
        @timer_with_info("Face crops", lambda result: f"Found {len(result)} faces")
        def detect_faces(video_frames):
            return face_detector.detect(video_frames)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = step_name or func.__name__.replace('_', ' ').title()

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time

            # Build log message
            log_msg = f"  ⏱ {name}: {duration:.2f}s"
            if info_func:
                try:
                    extra_info = info_func(result)
                    log_msg += f" | {extra_info}"
                except Exception:
                    pass  # Don't let info function break the main flow

            logging.info(log_msg)
            return result

        return wrapper

    return decorator