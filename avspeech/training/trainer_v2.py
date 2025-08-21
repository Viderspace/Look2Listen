# avspeech/training/trainer.py
from typing import Dict
from pathlib import Path
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, LambdaLR
# from torch.utils.tensorboard import SummaryWriter
from avspeech.training.training_phase import TrainingPhase, PhaseName
from avspeech.training.dataloader import MixedDataLoader
from avspeech.training.loss import ComplexCompressedLoss
from avspeech.utils.structs import SampleT

# Optional plotting helper (saved PNG per epoch)
try:
    from avspeech.training.metrics_viz import update_metrics_plot
except Exception:
    update_metrics_plot = None


class AVSpeechTrainer:
    def __init__(self, model: nn.Module, working_dir: Path, device: torch.device,
                 data_loader: MixedDataLoader, phase: TrainingPhase):
        self.model = model.to(device)
        self.device = device
        self.working_dir = working_dir
        self.data_loader = data_loader
        self.phase = phase
        self.criterion = ComplexCompressedLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.global_step = 0

        # Create checkpoint directory
        self.ckpt_dir = self.working_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Metrics logging / plotting outputs
        self.plots_dir = self.working_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = self.working_dir / "metrics.csv"
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, "w") as f:
                f.write("epoch,train_loss,val_loss,val_s2_noise,val_2s_clean")

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        # Placeholder for logging to TensorBoard or other tools
        pass

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        # Save model and optimizer state
        ckpt_path = self.ckpt_dir / f"model_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, ckpt_path)
        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
            }, best_path)

    def _setup_optimizer(self) -> Optimizer:
        """Create Adam optimizer with phase learning rate"""
        return torch.optim.Adam(
                self.model.parameters(),
                lr=self.phase.learning_rate
        )

    def _setup_scheduler(self) -> LRScheduler:
        """Paper-style LR schedule: halve at epochs 7 and 14 (step-based)."""
        steps_per_epoch = len(self.data_loader.sampler)
        print(f"{(steps_per_epoch == len(self.data_loader.get_train_loader())) =} equal?")
        total_steps = steps_per_epoch * self.phase.num_epochs

        # milestones by step (hard-coded at epoch 7 and 14 for a 20-epoch phase)
        m1 = 7 * steps_per_epoch
        m2 = 14 * steps_per_epoch

        # lightweight one-time print
        print(f"[sched] steps/epoch={steps_per_epoch}, total_steps={total_steps}, "
              f"lr_start={self.phase.learning_rate:.2e}, m1={m1}, m2={m2}")

        def lr_lambda(step: int) -> float:
            # piecewise-constant scale
            if step < m1:
                return 1.0
            elif step < m2:
                return 0.5
            else:
                return 0.25

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)



    def train(self):
        self.model.train()
        best_val_loss = float('inf')
        for epoch in range(self.phase.num_epochs):
            avg_loss = self.train_epoch(epoch)

            # Validation (always run each epoch)
            val_loss, val_s2_noise, val_2s_clean = self.run_validation(epoch)
            print(f"Validation Loss: {val_loss:.4f}")
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch + 1, is_best=True)

            # Append metrics and update plot
            try:
                with open(self.metrics_csv, "a") as f:
                    f.write(f"{epoch+1},{avg_loss:.6f},{val_loss:.6f},{'' if val_s2_noise is None else f'{val_s2_noise:.6f}'},{'' if val_2s_clean is None else f'{val_2s_clean:.6f}'}")
                if update_metrics_plot is not None:
                    out_png = self.plots_dir / "metrics.png"
                    update_metrics_plot(self.metrics_csv, out_png)
            except Exception as e:
                print(f"[metrics] logging/plot skipped: {e}")

            # Save checkpoint every epoch (non-best)
            self._save_checkpoint(epoch + 1, is_best=False)

    def train_epoch(self, epoch: int) -> float:
        """One epoch of training with running-average loss in tqdm."""
        epoch_loss = 0.0
        num_batches = 0

        train_loader = self.data_loader.get_train_loader()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.phase.num_epochs}")
        running_sum = 0.0
        count = 0

        for batch in progress:
            # audio_input, visual_input, target = batch
            # Move to device
            audio_input = batch['mixture'].to(self.device)
            target = batch['clean'].to(self.device)
            visual_input = batch['face'].to(self.device)



            audio_input = audio_input.to(self.device)
            visual_input = visual_input.to(self.device)
            target = target.to(self.device)

            # Forward pass
            output = self.model(audio_input, visual_input)
            loss = self.criterion(output, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.phase.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Running avg (per batch)
            running_sum += float(loss.detach().cpu())
            count += 1
            running_avg = running_sum / max(1, count)
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_avg:.4f}")

            # Log every 100 steps
            if self.global_step % 100 == 0:
                self._log_metrics({'train/loss': loss.item()}, self.global_step)

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1}, Average Train Loss: {avg_loss:.4f}")
        return avg_loss

    from typing import Dict, Tuple

    from typing import Tuple

    def run_validation(self, epoch: int) -> Tuple[float, float, float]:
        """
        Run validation over 2s_clean and 2s_noise, print per-split results (v1 style),
        and return a tuple in this exact order:
            (val_loss_overall, val_s2_noise, val_2s_clean)
        """
        print("Running full validation...")

        was_training = self.model.training
        self.model.eval()

        # 2s_clean
        s2c_avg, s2c_sum, s2c_batches = self._validate_split(SampleT.S2_CLEAN)
        if s2c_batches == 0:
            print(f"No validation data for {SampleT.S2_CLEAN.value}")
        else:
            print(f"  Val {SampleT.S2_CLEAN.value}: loss={s2c_avg:.4f}")

        # 2s_noise
        s2n_avg, s2n_sum, s2n_batches = self._validate_split(SampleT.S2_NOISE)
        if s2n_batches == 0:
            print(f"No validation data for {SampleT.S2_NOISE.value}")
        else:
            print(f"  Val {SampleT.S2_NOISE.value}: loss={s2n_avg:.4f}")

        total_batches = s2c_batches + s2n_batches
        overall = (s2c_sum + s2n_sum) / total_batches if total_batches > 0 else 0.0

        if was_training:
            self.model.train()

        # Return in requested order: (val_loss, val_s2_noise, val_2s_clean)
        return float(overall), float(s2n_avg), float(s2c_avg)


    def _validate_split(self, sample_type: SampleT) -> Tuple[float, float, int]:
        """
        Evaluate a single validation split; returns (avg_loss, total_loss, num_batches).
        Prints v1-style 'Validating ... with N batches...' message.
        """
        val_loader = self.data_loader.get_val_loader(sample_type)
        if not val_loader:
            return 0.0, 0.0, 0

        print(f"Validating {sample_type.value} with {len(val_loader)} batches...")

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                mixture = batch['mixture'].to(self.device)
                clean = batch['clean'].to(self.device)
                face = batch['face'].to(self.device)

                masks = self.model(mixture, face)
                separated = mixture * masks
                loss = self.criterion(separated, clean)

                total_loss += float(loss.item())
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, total_loss, num_batches