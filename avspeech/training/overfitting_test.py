from pathlib import Path
from typing import Tuple

from avspeech import DEVICE
from avspeech.training.training_phase import PhaseName
from avspeech.utils.structs import SampleT

path_to_samples = "/Users/jonatanvider/Desktop/Look2Listen_Stuff/overfitting_samples"


def setup():
    # Cell 6: Create phase configuration
    from avspeech.training.training_phase import TrainingPhase

    phase = TrainingPhase(
        name=PhaseName.MAIN,
        probabilities={
            SampleT.S2_NOISE: 1.00
        },
        num_epochs=100,
        learning_rate=1e-4,
        min_lr=2.5e-5,
            # learning_rate=6e-5,
            # min_lr=3e-5,
        batch_size=4,
        save_interval=100,
        gradient_clip=5.0
    )
    # phase.resume_checkpoint = str(resume_checkpoint)
    print(f"✓ Phase configured: {phase.name}")
    print(f"✓ Batch size: {phase.batch_size}")
    print(f"  Epochs: {phase.num_epochs}")
    print(f"  LR: {phase.learning_rate} -> {phase.min_lr}")
    print(f"  Mix: {phase.probabilities}")

    # Cell 7: Create dataloader
    from avspeech.training.dataloader import MixedDataLoader

    paths = {SampleT.S2_NOISE: Path(path_to_samples)}
    data_loader = MixedDataLoader(
            train_paths=paths,
            val_paths={},
            probabilities=phase.probabilities,
            batch_size=4,
            num_workers=1,
            seed=42
    )
    print(f"{data_loader.sampler.dataset_sizes=}")
    print(f"{len(data_loader.train_dataset)=}")
    print(f"✓ DataLoader created")
    print(f"  Batches per epoch: {len(data_loader.sampler)}")

    # Cell 8: Create model and trainer
    from avspeech.model.av_model import AudioVisualModel
    from avspeech.training.trainer import Trainer
    # from avspeech.training.trainer_v2 import AVSpeechTrainer


    model = AudioVisualModel()
    print(f"✓ Model created")

    # trainer = AVSpeechTrainer(
    #         model=model,
    #         phase=phase,
    #         data_loader=data_loader,
    #         device=DEVICE,
    #         working_dir= Path("/Users/jonatanvider/Documents/LookingToListenProject/av-speech-enhancement/avspeech/training/overfit_artifacts")
    # )
    trainer = Trainer(
            model=model,
            phase=phase,
            data_loader=data_loader,
            device=DEVICE,
            log_dir= Path("/Users/jonatanvider/Documents/LookingToListenProject/av-speech-enhancement/avspeech/training/overfit_artifacts"),
            checkpoint_dir=Path("/Users/jonatanvider/Documents/LookingToListenProject/av-speech-enhancement/avspeech/training/overfit_artifacts")

    )
    print(f"✓ Trainer initialized")

    # Cell 9: Start training
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    trainer.train()


def prin_epochs_loss():
    """
    Print the loss line only from the overfitting log file.
    """
    textfile_path = "/Users/jonatanvider/Documents/LookingToListenProject/av-speech-enhancement/debug/overfitting.txt"

    with open(textfile_path, "r") as file:
        lines = file.readlines()
    loss_lines = [line for line in lines if "loss=" in line]
    for line in loss_lines:
        print(line.strip())




if __name__ == "__main__":
    setup()
    # prin_epochs_loss()
    print("Overfitting test complete. Check ./overfit_checkpoints and ./overfit_logs for results.")