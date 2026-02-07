import gzip
import logging
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import config
from model import MLMModel  # type: ignore
from data import MLMDataset  # type: ignore


# Determine data directory - use /data if available, otherwise use path relative to script
def get_data_dir():
    data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    if not data_dir.exists():
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data"
    return data_dir


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_loader, device, checkpoint_dir="checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.optimizer = config.optimizer_class(
            self.model.parameters(), **config.optimizer_kwargs # type: ignore
        )

        self.scheduler = config.scheduler_class(
            self.optimizer, **config.scheduler_kwargs # type: ignore
        )

        self.global_step = 0
        self.best_loss = float("inf")
        self.warmup_steps = config.warmup_steps

        self.train_losses = []
        self.learning_rates = []

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def warmup_lr(self):
        scale = min(1.0, self.global_step / self.warmup_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = config.learning_rate * scale

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # ---- LR warmup (BEFORE optimizer step) ----
        if self.global_step < self.warmup_steps:
            self.warmup_lr()

        logits = self.model(input_ids)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # ---- Scheduler AFTER warmup ----
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()

        self.global_step += 1
        return loss.item()

    def save_checkpoint(self, name):
        path = self.checkpoint_dir / name
        torch.save(
            {
                "step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "losses": self.train_losses,
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["step"]
        self.train_losses = ckpt.get("losses", [])
        logger.info(f"Resumed from step {self.global_step}")

    def _format_time(self, seconds):
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def train(self, max_steps):
        self.model.train()
        logger.info(f"Training for {max_steps} steps")

        step_times = []
        train_start = time.time()

        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break

                step_start = time.time()
                loss = self.train_step(batch)
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                self.train_losses.append(loss)
                self.learning_rates.append(self.get_lr())

                if self.global_step % 100 == 0:
                    avg = np.mean(self.train_losses[-100:])
                    avg_time = np.mean(step_times[-100:])
                    remaining_steps = max_steps - self.global_step
                    eta_seconds = remaining_steps * avg_time
                    eta_str = self._format_time(eta_seconds)
                    
                    logger.info(
                        f"Step {self.global_step}/{max_steps} | "
                        f"Loss {avg:.4f} | LR {self.get_lr():.2e} | "
                        f"Time/Step {avg_time:.3f}s | ETA {eta_str}"
                    )

                    if avg < self.best_loss:
                        self.best_loss = avg
                        self.save_checkpoint("best_model.pt")

                if self.global_step % config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")

        total_time = time.time() - train_start
        self.save_checkpoint("final_model.pt")
        logger.info(f"Training complete in {self._format_time(total_time)}")


# ---------------- Data ----------------
def load_data(path, sample_size=None):
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                texts.append(line)
            if sample_size and i >= sample_size:
                break
    return texts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = get_data_dir()
    texts = load_data(str(data_dir / "corpus_clean.txt.gz"))

    dataset = MLMDataset(
        texts=texts,
        tokenizer_path="tokenizer.json",
        max_seq_len=config.max_seq_len,
        mask_rate_min=config.mask_rate_min,
        mask_rate_max=config.mask_rate_max,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=False,
    )

    model = MLMModel(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        ff_dim=config.ffn_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    trainer = Trainer(model, loader, device)

    checkpoints = sorted(trainer.checkpoint_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        trainer.load_checkpoint(checkpoints[-1])

    trainer.train(config.max_steps)


if __name__ == "__main__":
    main()