from pathlib import Path
from typing import Dict, Optional, Iterator, Any, List, Tuple

import torch
from torch.optim import Optimizer
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from avspeech.training.dataloader import MixedDataLoader
from avspeech.training.loss import ComplexCompressedLoss
from avspeech.utils.structs import SampleT
import avspeech.evaluation.metrics_bss as metrics_bss
from avspeech.training.training_phase import TrainingPhase


def _print_verbose(message: str, verbose: bool) -> None:
    if verbose:
        print(message)



# def log_metrics(metrics: Dict[str, float], step: int, log_dir : Path) -> None:
#     """Write metrics to tensorboard"""
#     writer = SummaryWriter(str(log_dir))
#
#     for key, value in metrics.items():
#         writer.add_scalar(key, value, step)
#
#     writer.close()


def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module , device : torch.device, verbose : bool = True) -> Dict[str, any]:
    """Load model and optimizer state from checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1


    _print_verbose(f"Loaded model weights from epoch {start_epoch}", verbose)

    return {
            'model': model,
            'epoch': start_epoch
    }


def save_checkpoint(epoch : int,
                    metrics: Dict[str, float],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    global_step: int,
                    checkpoint_dir: Path) -> None:
    """Save model, optimizer, and training state"""
    checkpoint = {
            "epoch"               : epoch,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step"         : global_step,
            "metrics"             : metrics,
    }

    path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, path)

    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def build_optimizer(parameters : Iterator[Any] , learning_rate: float) -> Optimizer:
    """Create Adam optimizer for the model"""
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    return optimizer

# def build_optimizer(parameters: Iterator[Any], learning_rate: float,
#                     weight_decay: float = 0.01,
#                     betas=(0.9, 0.999), eps: float = 1e-8) -> Optimizer:
#     """Create AdamW optimizer (single param group)."""
#     return torch.optim.AdamW(
#         parameters,
#         lr=learning_rate,
#         betas=betas,
#         eps=eps,
#         weight_decay=weight_decay
#     )



# def build_scheduler(
#         optimizer: Optimizer,
#         phase : TrainingPhase,
#         steps_per_epoch: int,
#         verbose: bool = True
# ) -> LRScheduler:
#     """Create cosine annealing LR scheduler"""
#     total_steps = steps_per_epoch * phase.num_epochs
#     _print_verbose(f"[sched] steps/epoch={steps_per_epoch}, total_steps={total_steps}, "
#               f"lr_start={phase.learning_rate:.2e}, lr_min={phase.min_lr:.2e}", verbose)
#
#     return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=phase.min_lr)

# scheduler.py

def build_scheduler(
    optimizer: Optimizer,
    phase,                 # needs: num_epochs, learning_rate, min_lr, warmup_fraction
    steps_per_epoch: int,
    verbose: bool = True,
) -> LRScheduler:
    total = steps_per_epoch * phase.num_epochs
    w = int(round(total * phase.warmup_fraction))          # warmup steps
    assert 0 <= w < total, "warmup_fraction must be in (0,1) and leave room for cosine"


    # linear warmup from 1e-2 * base_lr → base_lr
    warmup  = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=w)
    cosine  = CosineAnnealingLR(optimizer, T_max=total - w, eta_min=phase.min_lr)
    sched   = SequentialLR(optimizer, [warmup, cosine], milestones=[w])

    if verbose:
        _print_verbose(
            f"[sched] steps/epoch={steps_per_epoch}, total={total}, warmup={w} "
            f"({phase.warmup_fraction*100:.1f}%), T_max={total - w}, "
            f"lr_start={phase.learning_rate:.2e}, lr_min={phase.min_lr:.2e}",
            True
        )
    return sched


# def build_scheduler(
#     optimizer: Optimizer,
#     phase,                 # needs: num_epochs, learning_rate, min_lr, warmup_fraction
#     steps_per_epoch: int,
#     verbose: bool = True,
# ) -> LRScheduler:
#     total = steps_per_epoch * phase.num_epochs
#     w = int(round(total * phase.warmup_fraction))          # warmup steps
#     assert 0 <= w < total, "warmup_fraction must be in (0,1) and leave room for cosine"
#
#     scheduler = torch.optim.lr_scheduler.CyclicLR(
#             optimizer,
#             base_lr=phase.learning_rate,
#             max_lr=phase.learning_rate * 10,
#             step_size_up=4 * steps_per_epoch,
#             mode="triangular",
#             cycle_momentum=False,
#     )
#     return scheduler
#


def freeze_early_layers_on_warm_start(batch_idx: int, model : torch.nn.Module, unfreeze_batch: int) -> None:
    """Freeze first 2 audio conv layers at batch 0; unfreeze all at `unfreeze_batch`."""
    if batch_idx not in (0, unfreeze_batch):
        return

    def total_and_trainable_params_count():
        """Count trainable parameters in the audio CNN."""
        total = sum(p.numel() for p in model.audio_cnn.parameters())
        trainable = sum(p.numel() for p in model.audio_cnn.parameters() if p.requires_grad)
        return total, trainable

    # BEFORE
    total_params, trainable_before = total_and_trainable_params_count()
    action = "freeze" if batch_idx == 0 else "unfreeze"
    print(f"[{action}] before: total_params={total_params}, trainable={trainable_before}, step={batch_idx}")

    if batch_idx == 0:  # Freeze only the first 2 layers
        model.audio_cnn.freeze_early_layers(2)
    else:  # batch_idx == unfreeze_batch
        model.audio_cnn.unfreeze_all_layers()

    # AFTER
    total_params, trainable_after = total_and_trainable_params_count()
    print(f"[{action}] after: total_params={total_params}, trainable={trainable_after}")



def run_validation(device : torch.device,
                   model : torch.nn.Module,
                   data_manager: MixedDataLoader,
                   epoch: int,
                   log_dir: Path,
                   verbose: bool = True) -> None:
    """Run validation for all sample types"""
    # print("Running full validation...")
    for sample_type in (SampleT.S1_NOISE, SampleT.S2_CLEAN, SampleT.S2_NOISE):
        data_loader = data_manager.get_val_loader(sample_type)
        metrics = validate(device, model, data_loader, sample_type, verbose)
        if metrics:
            print(f"  Val {sample_type.value}: loss={metrics['loss']:.4f},  SDRi={metrics['sdri']:.4f}", flush=True)
            # log_metrics({f"val/{sample_type.value}": metrics["loss"]}, epoch, log_dir)





def validate(device : torch.device,
             model : torch.nn.Module,
             data_loader : Optional[DataLoader],
             sample_type : SampleT,
             verbose: bool = True) -> [str, float]:
    """Evaluate model on specific validation set"""
    if not data_loader:
        # _print_verbose(f"No validation data loader provided for {sample_type}", verbose)
        return {}

    model.eval()

    criterion = ComplexCompressedLoss()
    total_loss = 0
    num_batches = 0
    total_sdri = 0.0

    with torch.no_grad():
        for batch in data_loader:
            mixture = batch["mixture"].to(device)
            clean = batch["clean"].to(device)
            face = batch["face"].to(device)

            masks = model(mixture, face)
            separated = mixture * masks
            loss = criterion(separated, clean)

            total_sdri += evaluate_model_SDRI(separated, clean, mixture)
            total_loss += loss.item()
            num_batches += 1

    return {"loss": total_loss / num_batches if num_batches > 0 else 0, "sdri": total_sdri / num_batches if num_batches > 0 else 0.0}


@torch.no_grad()
def evaluate_model_SDRI(estimated_batch : torch.Tensor, gt_batch : torch.Tensor, mixture_batch : torch.Tensor ) -> float:
    """
    tensor shape: (Batch, C, T)

    """
    from avspeech.utils.fourier_transform import stft_to_audio

    sum_sdri = 0.0
    # print(f"{estimated_batch.shape=}, {gt_batch.shape=}")
    batch_size = estimated_batch.shape[0]

    for i in range(batch_size):
        estimated_sample = estimated_batch[i]
        clean_sample = gt_batch[i]
        mixture_sample = mixture_batch[i]

        estimated_wav = stft_to_audio(estimated_sample)
        clean_wav = stft_to_audio(clean_sample)
        mixture_wav = stft_to_audio(mixture_sample)
        sum_sdri += metrics_bss.sdri_bss(estimated_wav, clean_wav, mixture_wav)



    return sum_sdri / batch_size if batch_size > 0 else 0.0


def log_sdri_on_test_batch(test_batches: Dict[str, torch.Tensor], model: torch.nn.Module) -> str:
    """Evaluate SDRi on a fixed set of test batches."""
    with torch.no_grad():
        mixture, face, clean = test_batches["mixture"], test_batches["face"], test_batches["clean"]
        mask = model(mixture, face)
        separated = mixture * mask
        sdri = evaluate_model_SDRI(separated, clean, mixture)
        return f" SDRi={sdri:.4f} ,"






def log_grad_and_masks(model: torch.nn.Module, masks: torch.Tensor, step: Optional[int]=None) -> str:
    """Return a short string with grad L2 norm and mask stats."""
    with torch.no_grad():
        p05, p95 = torch.quantile(masks.detach().flatten(), masks.new_tensor([0.05, 0.95]))
        logits_std = torch.logit(masks.clamp(1e-6, 1-1e-6)).std()
# include: f"mask[p05={p05:.3f}, p95={p95:.3f}, spread={(p95-p05):.3f}] | logits_std={logits_std:.3f}"
        g2 = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g2 += p.grad.pow(2).sum().item()
        grad_l2 = g2 ** 0.5
        m = masks.detach()
        s = f"grad_l2={grad_l2:.4g} | mask[min={m.min().item():.3f}, p05={p05:.3f}, mean={m.mean().item():.3f}, p95={p95:.3f}, max={m.max().item():.3f} | spread={(p95-p05):.3f}] | logits_std={logits_std:.3f}"
        return f"step={step} | {s}" if step is not None else s


def grab_constant_batch(device: torch.device, data_loader: MixedDataLoader) -> Dict[str, torch.Tensor]:
    s2c_batch = next(iter(data_loader.get_val_loader(SampleT.S2_CLEAN)))

    mixture = s2c_batch["mixture"].to(device)
    clean = s2c_batch["clean"].to(device)
    face = s2c_batch["face"].to(device)

    return {"mixture": mixture, "face": face, "clean": clean}

