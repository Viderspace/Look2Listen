"""
Minimal Video Handler - Basic operations only
Focused on reliability and simplicity
"""

import subprocess
import json
import tempfile
import os
from pathlib import Path

def get_video_info(video_path):
    """
    Get basic video information using ffprobe

    Args:
        video_path: Path to video file

    Returns:
        dict with duration, width, height, fps
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Parse the data
        stream = data.get('streams', [{}])[0]
        format_data = data.get('format', {})

        # Parse frame rate (comes as fraction like "30/1")
        fps_str = stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return {
            'duration': float(format_data.get('duration', 0)),
            'width': stream.get('width', 0),
            'height': stream.get('height', 0),
            'fps': fps
        }
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
        raise Exception(f"Failed to get video info: {error_msg}")
    except Exception as e:
        raise Exception(f"Error parsing video info: {str(e)}")


def extract_segment(video_path, start_time, end_time, output_path=None):
    """
    Extract a segment from video with re-encoding for reliability

    Args:
        video_path: Input video path
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Optional output path (will create temp file if not provided)

    Returns:
        Path to extracted segment
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if end_time <= start_time:
        raise ValueError("End time must be after start time")

    # Create output path if not provided
    if output_path is None:
        Path("../temp").mkdir(exist_ok=True)
        output_path = tempfile.mktemp(suffix='.mp4', dir='../temp')

    duration = end_time - start_time

    # Use re-encoding for reliability (avoids keyframe issues)
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),      # Seek to start time
        '-i', video_path,            # Input file
        '-t', str(duration),         # Duration to extract
        '-c:v', 'libx264',          # Re-encode video with H.264
        '-preset', 'medium',         # Balance between speed and quality
        '-crf', '23',               # Good quality (lower = better, 23 is default)
        '-c:a', 'aac',              # Re-encode audio as AAC
        '-b:a', '128k',             # Audio bitrate
        '-movflags', '+faststart',  # Optimize for streaming
        '-y',                        # Overwrite output
        output_path
    ]

    try:
        # Run ffmpeg with timeout
        timeout = max(60, duration * 3)  # Timeout based on duration
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=True
        )

        # Verify output exists and has content
        if not os.path.exists(output_path):
            raise Exception("Output file was not created")

        if os.path.getsize(output_path) < 1000:  # Less than 1KB is suspicious
            raise Exception("Output file is too small, extraction may have failed")

        return output_path

    except subprocess.TimeoutExpired:
        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"Extraction timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        # Clean up failed file
        if os.path.exists(output_path):
            os.remove(output_path)
        error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
        raise Exception(f"FFmpeg failed: {error_msg}")

def cleanup_temp_files(temp_dir="temp", max_age_hours=1):
    """
    Clean up old temporary files

    Args:
        temp_dir: Directory to clean
        max_age_hours: Delete files older than this
    """
    import time

    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    cleaned = 0
    for file in temp_path.glob("*.mp4"):
        try:
            file_age = current_time - file.stat().st_mtime
            if file_age > max_age_seconds:
                file.unlink()
                cleaned += 1
        except Exception:
            pass  # File might be in use

    if cleaned > 0:
        print(f"Cleaned up {cleaned} old temporary files")

# Simple test
if __name__ == "__main__":
    print("Video handler module loaded successfully")
    print("Available functions:")
    print("  - get_video_info(video_path)")
    print("  - extract_segment(video_path, start_time, end_time)")
    print("  - cleanup_temp_files()")

    # Try to check if ffmpeg and ffprobe are available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg is available")
    except:
        print("❌ ffmpeg not found - please install ffmpeg")

    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        print("✅ ffprobe is available")
    except:
        print("❌ ffprobe not found - please install ffmpeg")