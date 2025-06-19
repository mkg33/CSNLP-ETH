#!/usr/bin/env python3

import argparse, os, random, time, json, sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from geomloss import SamplesLoss
from sklearn.metrics import f1_score


_next_checkpoint    = None
checkpoint_interval = None
ckpt_path           = None
results             = None

def save_checkpoint(path, model, optimizer, scheduler, blur, epoch, batch_idx, results_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "blur":           blur,
        "epoch":          epoch,
        "batch_idx":      batch_idx,
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "scheduler_state":scheduler.state_dict(),
        "results_so_far": results_dict
    }, path)
    print(f"[Checkpoint] blur={blur}, epoch={epoch}, batch={batch_idx} -> {path}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DataFrameDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        assert {"sentence1","sentence2","label"}.issubset(df.columns)
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["sentence1"], row["sentence2"], int(row["label"])


class OTStyleModel(nn.Module):
    def __init__(self,
             encoder_name: str = "microsoft/deberta-v3-base",
             hidden_dim: int = 768,
             content_dim: int = 192,
             style_dim: int = 128,
             lambda_orth: float = 1e-3,
             gate_init: float = 0.5,
             blur: float = 0.05,
             device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.lambda_orth = lambda_orth

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, add_prefix_space=True)
        self.encoder   = AutoModel.from_pretrained(encoder_name)

        self.content_proj = nn.Linear(self.encoder.config.hidden_size*2, content_dim)
        self.style_proj   = nn.Linear(self.encoder.config.hidden_size*2, style_dim)
        self.gate         = nn.Parameter(torch.tensor([gate_init]))

        feats = content_dim + style_dim + 1
        self.classifier = nn.Sequential(nn.LayerNorm(feats), nn.Linear(feats, 2))

        self.build_ot_loss(blur)

    def build_ot_loss(self, blur):
        self.ot = SamplesLoss("sinkhorn", p=1, blur=blur,
                              debias=True, potentials=False).to(self.device)
        self.blur = blur

    def tokenize(self, s1: List[str], s2: List[str], max_len=128):
        enc = self.tokenizer(s1, s2,
                             padding=True, truncation=True,
                             max_length=max_len,
                             return_tensors="pt",
                             return_offsets_mapping=True,
                             return_special_tokens_mask=True,
                             return_token_type_ids=False)
        tt = torch.zeros_like(enc["input_ids"])
        for i, mask in enumerate(enc["special_tokens_mask"]):
            sep = (enc["input_ids"][i] == self.tokenizer.sep_token_id)
            if sep.any():
                cut = sep.nonzero(as_tuple=False)[0,0].item() + 1
                tt[i, cut:] = 1
        enc["token_type_ids"] = tt
        enc.pop("offset_mapping"); enc.pop("special_tokens_mask")
        return {k: v.to(self.device) for k,v in enc.items()}

    def forward(self, s1, s2):
        enc = self.tokenize(s1, s2)
        out = self.encoder(**enc).last_hidden_state
        tt, am = enc["token_type_ids"], enc["attention_mask"]

        cls = out[:,0]
        def mmean(H, M):
            m = H * M.unsqueeze(-1)
            return m.sum(1)/(M.sum(1,True).clamp(min=1))
        m0 = mmean(out, (tt==0)&(am==1))
        m1 = mmean(out, (tt==1)&(am==1))

        rep = torch.cat([cls, m0-m1], dim=1)
        C   = self.content_proj(rep)
        S   = self.style_proj(rep)
        g   = torch.sigmoid(self.gate)
        fused = torch.cat([C*(1-g), S*g], dim=1)

        d_list = []
        for b in range(out.size(0)):
            i0 = ((tt[b]==0)&(am[b]==1)).nonzero(as_tuple=False)[:,0]
            i1 = ((tt[b]==1)&(am[b]==1)).nonzero(as_tuple=False)[:,0]
            d_list.append(self.ot(out[b,i0], out[b,i1]))
        d_ot = torch.stack(d_list).unsqueeze(1)

        feats = torch.cat([fused, d_ot], dim=1)
        logits= self.classifier(feats)


        Wc = self.content_proj.weight
        Ws = self.style_proj.weight
        ortho = (Wc @ Ws.T).pow(2).mean()

        return logits, ortho


def train_one_epoch(model, loader, optimizer, scheduler, ce_loss,
                    lambda_orth, blur, epoch, resume_batch=1):

    global _next_checkpoint, checkpoint_interval, ckpt_path, results
    model.train()
    total, correct, losses = 0, 0, 0.0

    for batch_idx, (s1, s2, y) in enumerate(loader, start=1):

        if batch_idx < resume_batch:
            continue

        now = time.time()
        if now >= _next_checkpoint:
            save_checkpoint(ckpt_path, model, optimizer, scheduler,
                            blur, epoch, batch_idx, results)
            _next_checkpoint = now + checkpoint_interval

        y_t = torch.tensor(y, device=model.device)
        optimizer.zero_grad()
        logits, ortho = model(s1, s2)
        loss = ce_loss(logits, y_t) + lambda_orth * ortho
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses += loss.item() * y_t.size(0)
        preds   = logits.argmax(dim=1)
        total  += y_t.size(0)
        correct+= (preds == y_t).sum().item()

    return losses / total, correct / total

@torch.no_grad()
def evaluate(model,loader,ce,lambda_orth):
    model.eval()
    ys, ps, ls = [], [], 0.0
    for s1,s2,y in loader:
        y_t = torch.tensor(y,device=model.device)
        logit, ortho = model(s1,s2)
        ls += (ce(logit,y_t)+lambda_orth*ortho).item()*y_t.size(0)
        ys+= y_t.cpu().tolist()
        ps+= logit.argmax(1).cpu().tolist()
    return ls/len(loader.dataset), f1_score(ys,ps,average="macro")


def main():
    global _next_checkpoint, checkpoint_interval, ckpt_path, results

    parser = argparse.ArgumentParser(
        description="Run blur sweep"
    )
    parser.add_argument("--train_features", type=str, required=True,
                        help="Pickle")
    parser.add_argument("--valid_features", type=str, required=True,
                        help="Pickle")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--warmup",      type=float, default=0.1,
                        help="Warmup for cosine")
    parser.add_argument("--content_dim", type=int,   default=192)
    parser.add_argument("--style_dim",   type=int,   default=128)
    parser.add_argument("--lambda_orth", type=float, default=1e-3)
    parser.add_argument("--device",      type=str,   default="cuda")
    parser.add_argument("--out_dir",     type=str,   default="runs/run1")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()


    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    checkpoint_interval = 600.0
    _next_checkpoint    = time.time() + checkpoint_interval
    ckpt_path           = os.path.join(args.out_dir, "periodic.ckpt.pt")


    if os.path.exists(ckpt_path):
        ckpt        = torch.load(ckpt_path, map_location="cpu")
        resume_blur = ckpt["blur"]
        start_epoch = ckpt["epoch"]
        resume_batch= ckpt["batch_idx"] + 1
        results     = ckpt["results_so_far"]
        print(f"Resuming from blur={resume_blur}, "
              f"epoch={start_epoch}, batch={resume_batch}")
    else:
        resume_blur = None
        start_epoch = 1
        resume_batch= 1
        results     = {}


    train_df = pd.read_pickle(args.train_features)
    valid_df = pd.read_pickle(args.valid_features)
    train_ds = DataFrameDataset(train_df)
    valid_ds = DataFrameDataset(valid_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: list(zip(*b))
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: list(zip(*b))
    )


    blur_grid = [0.01, 0.03, 0.05]
    ce_loss   = nn.CrossEntropyLoss()

    for blur in blur_grid:

        if resume_blur is not None and blur < resume_blur:
            continue

        print(f"\n> Starting blur = {blur:.3f} <")
        model = OTStyleModel(
            content_dim=args.content_dim,
            style_dim=args.style_dim,
            lambda_orth=args.lambda_orth,
            blur=blur,
            device=device
        ).to(device)

        optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = args.epochs * len(train_loader)
        scheduler   = get_cosine_schedule_with_warmup(
            optimizer,
            int(args.warmup * total_steps),
            total_steps
        )


        first_epoch = start_epoch if blur == resume_blur else 1
        best_f1     = results.get(blur, 0.0)

        for epoch in range(first_epoch, args.epochs + 1):

            rb = resume_batch if (blur == resume_blur and epoch == start_epoch) else 1

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader,
                optimizer, scheduler,
                ce_loss, args.lambda_orth,
                blur, epoch, rb
            )
            val_loss, val_f1 = evaluate(
                model, valid_loader,
                ce_loss, args.lambda_orth
            )

            print(f"[Blur={blur:.3f}] Ep {epoch} | "
                  f"tr_loss={tr_loss:.4f}, tr_acc={tr_acc:.3f} | "
                  f"val_loss={val_loss:.4f}, val_f1={val_f1:.3f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out_dir, f"best_blur{blur:.3f}.pt")
                )

        results[blur] = best_f1


    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Sweep done:", results)

if __name__=="__main__":
    main()
