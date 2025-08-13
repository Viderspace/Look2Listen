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
        #TODO - DEBUG VERSION REMOVE
        """Create cosine annealing LR scheduler"""
        steps_per_epoch = len(self.data_loader.sampler)
        print(f"{(steps_per_epoch == len(self.data_loader.get_train_loader())) =} equal?")
        total_steps = steps_per_epoch * self.phase.num_epochs

        # lightweight one-time print
        print(f"[sched] steps/epoch={steps_per_epoch}, total_steps={total_steps}, "
              f"lr_start={self.phase.learning_rate:.2e}, lr_min={self.phase.min_lr:.2e}")

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
        """Freeze first 2 audio conv layers at batch 0; unfreeze all at `unfreeze_batch`."""
        if batch_idx not in (0, unfreeze_batch):
            return
        def total_and_trainable_params_count():
            """Count trainable parameters in the audio CNN."""
            total = sum(p.numel() for p in self.model.audio_cnn.parameters())
            trainable = sum(p.numel() for p in self.model.audio_cnn.parameters() if p.requires_grad)
            return total, trainable

        # BEFORE
        total_params, trainable_before = total_and_trainable_params_count()
        action = "freeze" if batch_idx == 0 else "unfreeze"
        print(f"[{action}] before: total_params={total_params}, trainable={trainable_before}, step={batch_idx}")
        if batch_idx == 0:
            # Freeze only the first 2 layers
            self.model.audio_cnn.freeze_early_layers(2)
        else:  # batch_idx == unfreeze_batch
            self.model.audio_cnn.unfreeze_all_layers()

        # AFTER
        total_params, trainable_after = total_and_trainable_params_count()
        print(f"[{action}] after: total_params={total_params}, trainable={trainable_after}")


    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch and return average metrics"""
        self.model.train()
        train_loader = self.data_loader.get_train_loader()

        epoch_loss = 0
        num_batches = 0

        # ==========TODO - DEBUG REMOVE ===============================
        mix_counts_epoch = {"1s_noise": 0, "2s_clean": 0, "2s_noise": 0}
        _batch_mix_stats = []  # list of (n1, n2c, n2n) per batch
        # =========================================


        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            #==========TODO - DEBUG REMOVE ===============================
            # mix accounting per batch (cheap)
            types = batch["mix_type"]  # list[str]
            n1 = sum(1 for t in types if t == SampleT.S1_NOISE)
            n2c = sum(1 for t in types if t == SampleT.S2_CLEAN)
            n2n = sum(1 for t in types if t == SampleT.S2_NOISE)

            mix_counts_epoch["1s_noise"] += n1
            mix_counts_epoch["2s_clean"] += n2c
            mix_counts_epoch["2s_noise"] += n2n
            #=========================================
            #==========TODO - DEBUG REMOVE ===============================

            #=========================================

            # optional: light sample print (tune the interval to your device)
            if (self.global_step % 10) == 0:
                print(f"[mix/batch] 1S={n1} 2SC={n2c} 2SN={n2n}")

            _batch_mix_stats.append((n1, n2c, n2n))


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

        # ==========TODO - DEBUG REMOVE ===============================
        total_seen = sum(mix_counts_epoch.values())
        if total_seen > 0:
            p1 = mix_counts_epoch["1s_noise"] / total_seen
            p2c = mix_counts_epoch["2s_clean"] / total_seen
            p2n = mix_counts_epoch["2s_noise"] / total_seen
            try:
                import numpy as np

                arr = np.array(_batch_mix_stats, dtype=float)
                m0, m1, m2 = arr.mean(0).tolist()
                s0, s1, s2 = arr.std(0).tolist()
                print(f"[mix/epoch] seen={total_seen} "
                      f"pct=(1S={p1:.3f}, 2SC={p2c:.3f}, 2SN={p2n:.3f}) "
                      f"batch_mean=({m0:.1f},{m1:.1f},{m2:.1f}) "
                      f"batch_std=({s0:.1f},{s1:.1f},{s2:.1f})")
            except Exception:
                print(f"[mix/epoch] seen={total_seen} "
                      f"pct=(1S={p1:.3f}, 2SC={p2c:.3f}, 2SN={p2n:.3f})")
        # =========================================
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
            print(f"No validation data for {sample_type.value}")
            return {}

        print(f"Validating {sample_type.value} with {len(val_loader)} batches...")

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
