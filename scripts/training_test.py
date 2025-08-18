
from pathlib import Path
from avspeech.utils.constants import DEVICE
from avspeech.training.datasets import SampleT
from avspeech.model.av_model import AudioVisualModel
from avspeech.training.trainer import Trainer





def test():
    device = DEVICE
    datasets_path = Path('/Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets')

    output_dir = Path('/Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/output')

    val_set_paths = datasets_path / 'validation'
    train_set_paths = datasets_path / 'train'
    assert val_set_paths.exists(), f"Validation dataset not found: {val_set_paths}"
    assert train_set_paths.exists(), f"Training dataset not found: {train_set_paths}"

    train_paths = {SampleT.S1_NOISE: train_set_paths / '1s_noise', SampleT.S2_CLEAN: train_set_paths / '2s_clean', SampleT.S2_NOISE: train_set_paths / '2s_noise'}
    val_paths = {SampleT.S1_NOISE: val_set_paths / '1s_noise', SampleT.S2_CLEAN: val_set_paths / '2s_clean', SampleT.S2_NOISE: val_set_paths / '2s_noise'}

    resume_checkpoint = datasets_path / "checkpoint_epoch_23.pt"
    assert resume_checkpoint.exists(), f"Checkpoint not found: {resume_checkpoint}"
    print(f"✓ Paths configured")

    # Cell 6: Create phase configuration
    from avspeech.training.training_phase import TrainingPhase, PhaseName

    phase = TrainingPhase.warmstart(str(resume_checkpoint))
    phase = TrainingPhase(
            name=PhaseName.WARM_START,
            num_epochs=3,
            learning_rate=1e-4,
            min_lr=7e-5,
            batch_size=16,
            probabilities={SampleT.S1_NOISE: 0.9, SampleT.S2_CLEAN: 0.05, SampleT.S2_NOISE: 0.05},
            resume_checkpoint=str(resume_checkpoint)
    )

    # phase.resume_checkpoint = str(resume_checkpoint)
    print(f"✓ Phase configured: {phase.name}")
    print(f"  Epochs: {phase.num_epochs}")
    print(f"  LR: {phase.learning_rate} -> {phase.min_lr}")
    print(f"  Mix: {phase.probabilities}")

    # Cell 7: Create dataloader
    from avspeech.training.dataloader import MixedDataLoader

    for path in train_paths.values():
        print(path)

    print("actual val path : /Users/jonatanvider/Desktop/Look2Listen_Stuff/small_test_datasets/validation/1s_noise")
    for path in val_paths.values():
        print(path)

    data_loader = MixedDataLoader(
            train_paths=train_paths,
            val_paths=val_paths,
            probabilities=phase.probabilities,
            batch_size=phase.batch_size,
            num_workers=2,
            seed=42
    )

    print(f"{data_loader.sampler.dataset_sizes=}")
    print(f"{len(data_loader.train_dataset)=}")
    print(f"{len(data_loader.val_datasets)=}")
    print(f"✓ DataLoader created")
    print(f"  Batches per epoch: {len(data_loader.sampler)}")

    model = AudioVisualModel()
    print(f"✓ Model created")

    trainer = Trainer(
            model=model,
            phase=phase,
            data_loader=data_loader,
            device=device,
            checkpoint_dir=output_dir / 'checkpoints',
            log_dir=output_dir / 'logs',
    )
    print(f"✓ Trainer initialized")
    # Cell 9: Start training
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    trainer.train()



if __name__ == "__main__":
    test()
