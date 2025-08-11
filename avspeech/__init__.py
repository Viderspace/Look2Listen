# avspeech/__init__.py
"""
AV-Speech Enhancement Package
Visual-guided speech enhancement using lip movements
"""

__version__ = "0.1.0"

# Import main components for easier access
# Uncomment these as you implement them:

# from .inference import InferencePipeline
# from .model import AVSpeechModel
# from .preprocessing import VideoPreprocessor
# from .utils.face_embedder import FaceEmbedder

# Define what's available when someone does "from avspeech import *"
__all__ = [
    # "InferencePipeline",
    # "AVSpeechModel",
    # "VideoPreprocessor",
    # "FaceEmbedder",
]

# ================================================
# avspeech/utils/__init__.py
"""
Utility modules for AV-Speech Enhancement
"""

from .utils.face_embedder import FaceEmbedder
from .utils.video import *  # Import all video utilities
from .utils.constants import *  # Import all constants
from .utils.noise_mixer import NoiseMixer

__all__ = [
    "FaceEmbedder",
    "NoiseMixer",
    # Add other exports as needed
]

# ================================================
# avspeech/preprocessing/__init__.py
"""
Preprocessing modules for video and audio data
"""

from avspeech.preprocessing.clips_loader import *  # Or specific imports

__all__ = [
    # List exported classes/functions
]

# ================================================
# avspeech/model/__init__.py
"""
Model architecture and training utilities
"""

# from .architecture import AVSpeechModel
# from .trainer import Trainer

__all__ = [
    # "AVSpeechModel",
    # "Trainer",
]

# ================================================
# avspeech/inference/__init__.py
"""
Inference pipeline and post-processing
"""

# from .pipeline import InferencePipeline
# from .postprocess import AudioReconstructor

__all__ = [
    # "InferencePipeline",
    # "AudioReconstructor",
]