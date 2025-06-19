#!/usr/bin/env python3
import os
import sys
import time
import signal
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

#leftover from running in interactive mode
_should_terminate = False

def _on_sigusr1(signum, frame):
    global _should_terminate
    print("> SIGUSR1 received: exit plus checkpoint")
    _should_terminate = True

signal.signal(signal.SIGUSR1, _on_sigusr1)


def save_checkpoint(path, model, optimizer, queue, epoch, batch_idx):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "batch_idx":            batch_idx,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "queue": {
            "tensor": queue.queue.cpu(),
            "ptr":    queue.ptr.cpu()
        }
    }, path)
    print(f"Checkpoint saved: epoch {epoch}, batch {batch_idx}")


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive with checkpoints")
    p.add_argument("--train-features", type=str, required=True,
                   help="Path to pickle with sentence1, sentence2, label")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Dir to save checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--proj-dim", type=int, default=768)
    p.add_argument("--queue-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--print-freq", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


class PrecomputedDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 128):
        df = pd.read_pickle(path)
        for col in ("sentence1", "sentence2", "label"):
            if col not in df.columns:
                raise KeyError(f"Column {col!r} missing in {path}")
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        s1, s2 = row["sentence1"], row["sentence2"]
        y = int(row["label"])
        t1 = self.tokenizer(
            s1, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        t2 = self.tokenizer(
            s2, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "ids1":  t1["input_ids"].squeeze(0),
            "mask1": t1["attention_mask"].squeeze(0),
            "ids2":  t2["input_ids"].squeeze(0),
            "mask2": t2["attention_mask"].squeeze(0),
            "label": torch.tensor(y, dtype=torch.long),
        }


class StyleEncoder(nn.Module):
    def __init__(self, proj_dim: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("roberta-base")
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        z = self.proj(h)
        return F.normalize(z, dim=1)


class CLQueue(nn.Module):
    def __init__(self, dim: int, K: int):
        super().__init__()
        self.register_buffer("queue", F.normalize(torch.randn(K, dim), dim=1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.K = K

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        B, K = keys.size(0), self.K
        ptr = int(self.ptr)
        if B > K:
            keys = keys[-K:]
            B = K
        end = ptr + B
        if end <= K:
            self.queue[ptr:end] = keys.detach()
        else:
            first = K - ptr
            self.queue[ptr:]       = keys[:first].detach()
            self.queue[:end - K]   = keys[first:].detach()
        self.ptr[0] = end % K

    def get(self) -> torch.Tensor:
        return self.queue


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    tau: torch.Tensor,
    anchor_count: int
) -> torch.Tensor:
    sim = z @ z.t() / tau
    sim.masked_fill_(torch.eye(len(z), device=z.device).bool(), float("-inf"))
    valid = labels >= 0
    pos = (labels[:, None] == labels[None, :]) & valid[:, None] & valid[None, :]
    sim_a, pos_a = sim[:anchor_count], pos[:anchor_count]
    denom = torch.logsumexp(sim_a, dim=1)
    has = pos_a.any(1)
    if not has.any():
        return torch.tensor(0., device=z.device)
    num = torch.logsumexp(sim_a.masked_fill(~pos_a, float("-inf")), dim=1)
    return -(num[has] - denom[has]).mean()


def train_one_epoch(
    model: StyleEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    queue: CLQueue,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    resume_batch: int,
    ckpt_path: str,
    checkpoint_interval: float
) -> None:
    global _should_terminate, _next_checkpoint
    model.train()
    total_loss, steps = 0.0, 0

    for batch_idx, batch in enumerate(loader, start=1):
        # Skip to resume
        if epoch == _start_epoch and batch_idx < resume_batch:
            continue


        ids  = torch.cat([batch["ids1"], batch["ids2"]], dim=0).to(device)
        mask = torch.cat([batch["mask1"], batch["mask2"]], dim=0).to(device)
        y    = batch["label"].to(device)

        # positives
        keep = (y == 0).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            continue
        orig_B = y.size(0)
        idxs   = torch.cat([keep, keep + orig_B], dim=0)
        ids_k, mask_k = ids[idxs], mask[idxs]


        with autocast():
            z    = model.encode(ids_k, mask_k)
            Bpos = keep.size(0)
            labs = torch.cat([torch.arange(Bpos, device=device)] * 2, dim=0)
            q    = queue.get()
            all_z   = torch.cat([z, q], dim=0)
            all_lbl = torch.cat([
                labs,
                torch.full((q.size(0),), -1, dtype=torch.long, device=device)
            ], dim=0)
            loss = supervised_contrastive_loss(all_z, all_lbl, model.tau(), 2 * Bpos)


        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        queue.enqueue(z.detach())

        total_loss += loss.item()
        steps += 1

        now = time.time()
        if now >= _next_checkpoint:
            save_checkpoint(ckpt_path, model, optimizer, queue, epoch, batch_idx)
            _next_checkpoint = now + checkpoint_interval

        if _should_terminate:
            save_checkpoint(ckpt_path, model, optimizer, queue, epoch, batch_idx)
            print("Exiting early after checkpoint.")
            sys.exit(0)

        if batch_idx % args.print_freq == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx}/{len(loader)}"
                  f" Loss={loss.item():.4f} τ={model.tau().item():.4f}")


    save_checkpoint(ckpt_path, model, optimizer, queue, epoch, batch_idx)
    avg_loss = total_loss / max(1, steps)
    print(f"[Epoch {epoch} done] AvgLoss={avg_loss:.4f} τ={model.tau().item():.4f}")


def main():
    global _start_epoch, _next_checkpoint, args
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")


    ds = PrecomputedDataset(args.train_features, tokenizer)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )


    model = StyleEncoder(args.proj_dim).to(device)
    queue = CLQueue(dim=args.proj_dim, K=args.queue_size).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()


    ckpt_path = os.path.join(args.output_dir, "contrastive_checkpoint.pt")
    _start_epoch = 1
    resume_batch = 1

    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        queue.queue.copy_(ckpt["queue"]["tensor"])
        queue.ptr.copy_(ckpt["queue"]["ptr"])
        _start_epoch = ckpt.get("epoch", 1)
        resume_batch = ckpt.get("batch_idx", 0) + 1
        print(f"Resuming from epoch {_start_epoch}, batch {resume_batch}")


    checkpoint_interval = 600.0
    _next_checkpoint = time.time() + checkpoint_interval


    for epoch in range(_start_epoch, args.epochs + 1):
        train_one_epoch(
            model, loader, optimizer, queue, scaler,
            device, epoch, resume_batch,
            ckpt_path, checkpoint_interval
        )
        resume_batch = 1

    done_file = os.path.join(args.output_dir, "TRAIN_COMPLETE")
    with open(done_file, "w") as f:
        f.write("done\n")
    print("Training complete.")

if __name__ == "__main__":
    main()
