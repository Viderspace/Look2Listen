# avspeech/training/trainer.py
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from avspeech.training.dataloader import MixedDataLoader
from avspeech.training.loss import ComplexCompressedLoss
from avspeech.training.training_phase import PhaseName, TrainingPhase
import avspeech.training.trainer_tools as tools

print_verbose = False  # Set to False to disable verbose logging


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        phase: TrainingPhase,
        data_loader: MixedDataLoader,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
    ):
        """Initialize trainer with model, phase config, and paths"""
        self.model = model.to(device)
        self.phase = phase
        self.data_loader = data_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.criterion = ComplexCompressedLoss()
        self.optimizer = tools.build_optimizer(self.model.parameters(), phase.learning_rate)
        self.scheduler = tools.build_scheduler(self.optimizer, phase, len(data_loader.get_train_loader()), print_verbose)

        self.start_epoch = 1
        self.global_step = 0

        if phase.resume_checkpoint:
            self._load_checkpoint(phase.resume_checkpoint)


    def train(self) -> None:
        """Main training loop for all epochs"""
        print(f"Starting {self.phase.name} phase: {self.phase.num_epochs} epochs")

        for epoch in range(self.start_epoch, self.start_epoch + self.phase.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}")

            # Validate
            if epoch % self.phase.val_interval == 0:  # usually every epoch
                self._run_full_validation(epoch)
            # Save
            if epoch % self.phase.save_interval == 0:  # usually every epoch
                self._save_checkpoint(epoch, train_metrics)

        print("Training complete")


    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch and return average metrics"""
        self.model.train()
        train_loader = self.data_loader.get_train_loader()

        accumulated_loss = 0
        how_many_batches_done = 0

        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.phase.num_epochs}")

        for batch in tqdm_bar:
            # Freeze early layers only on the first epoch of 'warm start' phase
            if epoch == self.start_epoch and self.phase.name == PhaseName.WARM_START:
                tools.freeze_early_layers_on_warm_start(how_many_batches_done, self.model, len(train_loader) // 2)

            batch_loss = self.train_batch(batch)

            # Track metrics
            accumulated_loss += batch_loss
            how_many_batches_done += 1

            running_avg = accumulated_loss / max(1, how_many_batches_done)
            tqdm_bar.set_postfix(recent_loss=f"{batch_loss:.4f}", current_avg=f"{running_avg:.4f}")

            # Log every 100 steps
            # if self.global_step % 500 == 0:
            #     tools.log_metrics({"train/loss": batch_loss}, self.global_step, self.log_dir)

        return {"loss": accumulated_loss / how_many_batches_done}


    def train_batch(self, batch : Dict[str, Any]) -> float:
        # Move to device
        mixture = batch["mixture"].to(self.device)
        clean = batch["clean"].to(self.device)
        face = batch["face"].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        masks = self.model(mixture, face)
        separated = mixture * masks
        loss = self.criterion(separated, clean)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.phase.gradient_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return loss.item()


# =================     Utilities / Wrappers    ===========================
    def _run_full_validation(self, epoch: int) -> None:
        """Run validation for all sample types"""
        tools.run_validation(
                device=self.device,
                model=self.model,
                data_manager=self.data_loader,
                epoch=epoch,
                log_dir=self.log_dir,
                verbose=print_verbose
        )


    def _load_checkpoint(self, checkpoint_path: str) -> None:
        result = tools.load_checkpoint(
                Path(checkpoint_path), self.model, self.device, verbose=print_verbose)
        self.model = result['model']
        self.start_epoch = result['epoch']


    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        tools.save_checkpoint(
                epoch=epoch,
                metrics=metrics,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                global_step=self.global_step,
                checkpoint_dir=self.checkpoint_dir
        )
