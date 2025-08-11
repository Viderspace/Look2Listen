# avspeech/training/trainer.py
from typing import Dict
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from avspeech.training.training_phase import TrainingPhase, PhaseName
from avspeech.training.dataloader import MixedDataLoader
from avspeech.training.loss import ComplexCompressedLoss
from avspeech.utils.structs import SampleT


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            phase: TrainingPhase,
            data_loader: MixedDataLoader,
            device: torch.device,
            checkpoint_dir: Path,
            log_dir: Path
    ):
        """Initialize trainer with model, phase config, and paths"""
        self.model = model.to(device)
        self.phase = phase
        self.data_loader = data_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.criterion = ComplexCompressedLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.writer = SummaryWriter(str(log_dir))

        self.start_epoch = 1
        self.global_step = 0

        if phase.resume_checkpoint:
            self._load_checkpoint(phase.resume_checkpoint)

    def _setup_optimizer(self) -> Optimizer:
        """Create Adam optimizer with phase learning rate"""
        return torch.optim.Adam(
                self.model.parameters(),
                lr=self.phase.learning_rate
        )

    def _setup_scheduler(self) -> LRScheduler:
        """Create cosine annealing LR scheduler"""
        steps_per_epoch = len(self.data_loader.sampler)
        total_steps = steps_per_epoch * self.phase.num_epochs

        return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.phase.min_lr
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and restore model/optimizer states"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)

        print(f"Resumed from epoch {checkpoint['epoch']}")

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model, optimizer, and training state"""
        checkpoint = {
                'epoch'               : epoch,
                'model_state_dict'    : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'global_step'         : self.global_step,
                'metrics'             : metrics,
                'phase'               : self.phase.name.value
        }

        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)

        # Also save as latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)

    def freeze_early_layers_on_warm_start(self, batch_idx: int, unfreeze_batch: int) -> None:
        if batch_idx == 0:
            self.model.audio_cnn.freeze_early_layers(2)  # Freeze only the first 2 layers of the audio CNN
        elif batch_idx == unfreeze_batch:  # Unfreeze all layers after the specified batch
            self.model.audio_cnn.unfreeze_all_layers()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch and return average metrics"""
        self.model.train()
        train_loader = self.data_loader.get_train_loader()

        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):

            # Freeze early layers only on the first epoch of warm start
            if epoch == self.start_epoch and self.phase.name == PhaseName.WARM_START:
                self.freeze_early_layers_on_warm_start(batch_idx=num_batches, unfreeze_batch=len(train_loader) // 2)

            # Move to device
            mixture = batch['mixture'].to(self.device)
            clean = batch['clean'].to(self.device)
            face = batch['face'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            masks = self.model(mixture, face)
            separated = mixture * masks
            loss = self.criterion(separated, clean)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.phase.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log every 100 steps
            if self.global_step % 100 == 0:
                self._log_metrics({'train/loss': loss.item()}, self.global_step)

        return {'loss': epoch_loss / num_batches}


    def train(self) -> None:
        """Main training loop for all epochs"""
        print(f"Starting {self.phase.name} phase: {self.phase.num_epochs} epochs")

        for epoch in range(self.start_epoch, self.start_epoch + self.phase.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}")

            # Validate
            if epoch % self.phase.val_interval == 0:  # usually every epoch
                self.run_full_validation(epoch)
            # Save
            if epoch % self.phase.save_interval == 0:  # usually every epoch
                self._save_checkpoint(epoch, train_metrics)

        self.writer.close()
        print(f"Training complete")

    def run_full_validation(self, epoch : int) -> None:
        """Run validation for all sample types"""
        print("Running full validation...")
        for sample_type in [SampleT.S1_NOISE, SampleT.S2_CLEAN, SampleT.S2_NOISE]:
            metrics = self.validate(sample_type)
            if metrics:
                print(f"  Val {sample_type.value}: loss={metrics['loss']:.4f}")
                self._log_metrics({f'val/{sample_type.value}': metrics['loss']}, epoch)



    def validate(self, sample_type: SampleT) -> Dict[str, float]:
        """Evaluate model on specific validation set"""
        self.model.eval()
        val_loader = self.data_loader.get_val_loader(sample_type)

        if not val_loader:
            return {}

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                mixture = batch['mixture'].to(self.device)
                clean = batch['clean'].to(self.device)
                face = batch['face'].to(self.device)

                masks = self.model(mixture, face)
                separated = mixture * masks
                loss = self.criterion(separated, clean)

                total_loss += loss.item()
                num_batches += 1

        return {'loss': total_loss / num_batches if num_batches > 0 else 0}

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Write metrics to tensorboard"""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
