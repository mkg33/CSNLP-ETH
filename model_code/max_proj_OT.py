# maximum projection OT version
import argparse, os, random, time, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DataFrameDataset(Dataset):
    REQ = {"sentence1", "sentence2", "label"}

    def __init__(self, df: pd.DataFrame):
        if not self.REQ.issubset(df.columns):
            raise ValueError(f"DataFrame missing {self.REQ - set(df.columns)}")
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return row["sentence1"], row["sentence2"], int(row["label"])



class OTStyleModel(nn.Module):
    def __init__(
        self,
        encoder: str = "microsoft/deberta-v3-base",
        content_dim: int = 192,
        style_dim: int = 128,
        lambda_orth: float = 1e-3,
        #lambda_orth: float = 0,
        n_proj: int = 128,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.lambda_orth = lambda_orth
        self.n_proj = n_proj
        self.tokenizer = AutoTokenizer.from_pretrained(encoder, add_prefix_space=True)
        self.encoder = AutoModel.from_pretrained(encoder)

        hidden2 = self.encoder.config.hidden_size * 2
        self.content_proj = nn.Linear(hidden2, content_dim)
        self.style_proj = nn.Linear(hidden2, style_dim)
        self.gate = nn.Parameter(torch.tensor([0.5]))


        self.classifier = nn.Sequential(
            nn.LayerNorm(content_dim + style_dim + 1),
            nn.Linear(content_dim + style_dim + 1, 2)
        )

    @staticmethod
    def max_sliced_w1(x: torch.Tensor, y: torch.Tensor,
                      n_proj: int = 128) -> torch.Tensor:
        """
        (Monte-Carlo) max-sliced Wasserstein-1 between two point clouds.
        Returns: one scalar.
        """
        d = x.size(1)
        dirs = torch.randn(n_proj, d, device=x.device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)           # unit vectors
        xp, _ = (x @ dirs.t()).sort(dim=0)
        yp, _ = (y @ dirs.t()).sort(dim=0)
        m = min(xp.size(0), yp.size(0))
        diff = torch.abs(xp[:m] - yp[:m]).mean(0)
        return diff.max()


    def _tokenize(self, s1, s2, max_len: int = 128):
        enc = self.tokenizer(
            s1,
            s2,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
        # build token-type ids
        tt = torch.zeros_like(enc["input_ids"])
        for i, mask in enumerate(enc["special_tokens_mask"]):
            sep = enc["input_ids"][i] == self.tokenizer.sep_token_id
            if sep.any():
                tt[i, sep.nonzero(as_tuple=False)[0, 0] + 1 :] = 1
        enc["token_type_ids"] = tt
        enc.pop("offset_mapping")
        enc.pop("special_tokens_mask")
        print("Tokenize...")
        return {k: v.to(self.device) for k, v in enc.items()}


    def forward(self, s1, s2):
        inp = self._tokenize(s1, s2)
        H = self.encoder(**inp).last_hidden_state
        tt, m = inp["token_type_ids"], inp["attention_mask"]
        cls = H[:, 0]

        print("Forward now...")

        def mmean(h, mask):
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True
            ).clamp(min=1)

        mean0 = mmean(H, (tt == 0) & (m == 1))
        mean1 = mmean(H, (tt == 1) & (m == 1))
        rep = torch.cat([cls, mean0 - mean1], 1)

        C = self.content_proj(rep)
        S = self.style_proj(rep)
        g = torch.sigmoid(self.gate)
        fused = torch.cat([C * (1 - g), S * g], 1)

        dists = []
        for b in range(H.size(0)):
            a_idx = ((tt[b] == 0) & (m[b] == 1)).nonzero(as_tuple=True)[0]
            b_idx = ((tt[b] == 1) & (m[b] == 1)).nonzero(as_tuple=True)[0]
            # token clouds
            dists.append(
                self.max_sliced_w1(
                    H[b, a_idx],
                    H[b, b_idx],
                    n_proj=self.n_proj
                )
            )
        d_ot = torch.stack(dists).unsqueeze(1)

        logits = self.classifier(torch.cat([fused, d_ot], 1))

        Wc, Ws = self.content_proj.weight, self.style_proj.weight
        ortho = (Wc @ Ws.T).pow(2).mean()
        return logits, ortho


def train_one_epoch(model, loader, opt, sched, ce, lam):
    model.train()
    tot, corr, loss_sum = 0, 0, 0.0
    print("Starting epoch...")
    i = 1
    for s1, s2, y in loader:
        y_t = torch.tensor(y, device=model.device, dtype=torch.long)
        opt.zero_grad()
        logits, ortho = model(s1, s2)
        loss = ce(logits, y_t) + lam * ortho
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_sum += loss.item() * y_t.size(0)
        tot += y_t.size(0)
        corr += (logits.argmax(1) == y_t).sum().item()
        print("Loss: ")
        print(loss_sum)
        print("Epoch: ")
        print(i)
    i+=1
    return loss_sum / tot, corr / tot

@torch.no_grad()
def evaluate(model, loader, ce, lam):
    model.eval()
    ys, ps, ls = [], [], 0.0
    for s1, s2, y in loader:
        y_t = torch.tensor(y, device=model.device, dtype=torch.long)
        logits, ortho = model(s1, s2)
        ls += (ce(logits, y_t) + lam * ortho).item() * y_t.size(0)
        ys.extend(y_t.cpu())
        ps.extend(logits.argmax(1).cpu())
    return ls / len(loader.dataset), f1_score(ys, ps, average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--valid_features", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--content_dim", type=int, default=192)
    parser.add_argument("--style_dim", type=int, default=128)
    parser.add_argument("--lambda_orth", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_proj", type=int, default=128,
                    help="number of random directions for max-sliced OT")
    parser.add_argument("--out_dir", type=str, default="runs/sliced_ot_no_ckpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    tr_df = pd.read_pickle(args.train_features)
    va_df = pd.read_pickle(args.valid_features)
    tr_loader = DataLoader(
        DataFrameDataset(tr_df),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: list(zip(*b)),
    )
    va_loader = DataLoader(
        DataFrameDataset(va_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: list(zip(*b)),
    )


    model = OTStyleModel(
        content_dim=args.content_dim,
        style_dim=args.style_dim,
        n_proj=args.n_proj,
        lambda_orth=args.lambda_orth,
        device=device,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(
        opt,
        int(args.warmup * args.epochs * len(tr_loader)),
        args.epochs * len(tr_loader),
    )
    ce = nn.CrossEntropyLoss()

    best_f1 = 0.0


    for ep in range(1, args.epochs + 1):
        print(args.lambda_orth)
        tr_loss, tr_acc = train_one_epoch(
            model, tr_loader, opt, sched, ce, args.lambda_orth
        )
        va_loss, va_f1 = evaluate(model, va_loader, ce, args.lambda_orth)
        print(
            f"Ep{ep:02d}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
            f"va_loss={va_loss:.4f} va_f1={va_f1:.3f}"
        )
        best_f1 = max(best_f1, va_f1)


    torch.save(model.state_dict(), os.path.join(args.out_dir, "max_proj.pt"))
    json.dump(
        {"best_f1": best_f1},
        open(os.path.join(args.out_dir, "results.json"), "w"),
        indent=2,
    )
    print("Training done. Final validation F1 =", best_f1)


if __name__ == "__main__":
    main()
