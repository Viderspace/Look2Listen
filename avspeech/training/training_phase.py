# avspeech/training/phase.py
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from avspeech.utils.structs import SampleT


class PhaseName(Enum):
    """Enum for training phases"""

    WARM_START = "Warm start"
    MAIN = "Main"
    POLISH = "Polish"


@dataclass
class TrainingPhase:
    """Configuration for a training phase"""

    name: PhaseName
    probabilities: Dict[SampleT, float]
    num_epochs: int
    learning_rate: float
    min_lr: float
    batch_size: int = 8
    gradient_clip: float = 1.0
    save_interval: int = 1
    val_interval: int = 1
    resume_checkpoint: Optional[str] = None

    @classmethod
    def warmstart(cls, checkpoint_path: Optional[str] = None) -> "TrainingPhase":
        return cls(
            name=PhaseName.WARM_START,
            probabilities={
                SampleT.S1_NOISE: 0.05,
                SampleT.S2_CLEAN: 0.475,
                SampleT.S2_NOISE: 0.475,
            },
            num_epochs=4,
            learning_rate=6e-5,
            min_lr=3e-5,
            resume_checkpoint=checkpoint_path,
        )

    @classmethod
    def main(cls, checkpoint_path: Optional[str] = None) -> "TrainingPhase":
        return cls(
            name=PhaseName.MAIN,
            # probabilities={SampleT.S1_NOISE: 0.10, SampleT.S2_CLEAN: 0.45, SampleT.S2_NOISE: 0.45},
            probabilities={
                SampleT.S1_NOISE: 0.08,
                SampleT.S2_CLEAN: 0.46,
                SampleT.S2_NOISE: 0.46,
            },
            num_epochs=8,
            learning_rate=3e-5,
            min_lr=8e-6,
            resume_checkpoint=checkpoint_path,
        )

    @classmethod
    def e_25_resume(cls, e25_checkpoint_path: Optional[str] = None) -> "TrainingPhase":
        return cls(
            name=PhaseName.MAIN,
            probabilities={
                SampleT.S1_NOISE: 0.08,
                SampleT.S2_CLEAN: 0.46,
                SampleT.S2_NOISE: 0.46,
            },
            num_epochs=6,
            learning_rate=2e-5,
            min_lr=5e-6,
            resume_checkpoint=e25_checkpoint_path,
        )

    @classmethod
    def polish(cls, checkpoint_path: Optional[str] = None) -> "TrainingPhase":
        return cls(
            name=PhaseName.POLISH,
            probabilities={
                SampleT.S1_NOISE: 0.10,
                SampleT.S2_CLEAN: 0.45,
                SampleT.S2_NOISE: 0.45,
            },
            num_epochs=5,
            learning_rate=3e-5,
            min_lr=1e-5,
            resume_checkpoint=checkpoint_path,
        )
