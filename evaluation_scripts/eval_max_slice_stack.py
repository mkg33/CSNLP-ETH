#!/usr/bin/env python3
"""
Eval for max-sliced OTStyleModel (the StackExchange dataset).
All experiments have been run with seed = 42.
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def max_sliced_w1_proj(x: torch.Tensor, y: torch.Tensor,
                       proj: torch.Tensor) -> torch.Tensor:
    xp, _ = (x @ proj.T).sort(dim=0)
    yp, _ = (y @ proj.T).sort(dim=0)
    m = min(xp.size(0), yp.size(0))
    return (xp[:m] - yp[:m]).abs().mean(0).max()



class OTStyleModel(nn.Module):
    def __init__(self,
                 encoder: str              = "microsoft/deberta-v3-base",
                 content_dim: int          = 192,
                 style_dim: int            = 128,
                 n_proj: int               = 128,
                 proj_seed: int            = 42,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)


        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, add_prefix_space=True)
        self.encoder = AutoModel.from_pretrained(encoder)

        hidden2 = self.encoder.config.hidden_size * 2
        self.content_proj = nn.Linear(hidden2, content_dim)
        self.style_proj   = nn.Linear(hidden2, style_dim)
        self.gate         = nn.Parameter(torch.tensor([0.5]))

        self.classifier = nn.Sequential(
            nn.LayerNorm(content_dim + style_dim + 1),
            nn.Linear(content_dim + style_dim + 1, 2),
        )


        g = torch.Generator(device).manual_seed(proj_seed)
        dirs = torch.randn(n_proj, hidden2 // 2, generator=g,
                           device=self.device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        self.register_buffer("proj_dirs", dirs)


    def _tokenize(self, s1: List[str], s2: List[str],
                  max_len: int = 128):
        enc = self.tokenizer(
            s1, s2,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}


    def forward(self, s1: List[str], s2: List[str]) -> torch.Tensor:
        inp = self._tokenize(s1, s2)
        H   = self.encoder(**inp).last_hidden_state
        tt, m = inp["token_type_ids"], inp["attention_mask"]
        cls = H[:, 0]

        def masked_mean(h, mask):
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True).clamp(min=1)

        mean0 = masked_mean(H, (tt == 0) & (m == 1))
        mean1 = masked_mean(H, (tt == 1) & (m == 1))
        rep   = torch.cat([cls, mean0 - mean1], dim=1)


        C = self.content_proj(rep)
        S = self.style_proj(rep)
        g = torch.sigmoid(self.gate)
        fused = torch.cat([C * (1 - g), S * g], dim=1)


        dists = []
        for b in range(H.size(0)):
            a_idx = ((tt[b] == 0) & (m[b] == 1)).nonzero(as_tuple=True)[0]
            b_idx = ((tt[b] == 1) & (m[b] == 1)).nonzero(as_tuple=True)[0]
            dists.append(max_sliced_w1_proj(H[b, a_idx], H[b, b_idx],
                                             self.proj_dirs))
        d_ot = torch.stack(dists).unsqueeze(1)

        logits = self.classifier(torch.cat([fused, d_ot], dim=1))
        return logits


def load_model(ckpt: Path, device: torch.device,
               n_proj: int, seed: int) -> OTStyleModel:
    sd = torch.load(ckpt, map_location=device)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    c_dim = sd["content_proj.weight"].shape[0]
    s_dim = sd["style_proj.weight"].shape[0]

    model = OTStyleModel(content_dim=c_dim,
                         style_dim=s_dim,
                         n_proj=n_proj,
                         proj_seed=seed,
                         device=device)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "max-sliced OT model "
        "(StackExchange)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="Model (.pt)")
    p.add_argument("--input-dir", required=True,
                   help="dir with problem-*.txt docs")
    p.add_argument("--output-dir", required=True,
                   help="Output for solutions.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--n-proj", type=int, default=128,
                   help="Num. of random projections (must match training!!!)")
    p.add_argument("--seed", type=int, default=42,
                   help="All experiments have been run with 42.")
    return p.parse_args()


@torch.no_grad()
def predict_dir(model: OTStyleModel,
                inp_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in tqdm(sorted(inp_dir.glob("problem-*.txt")),
                         unit="file"):
        pid = txt_file.stem[8:]
        with txt_file.open(encoding="utf-8") as fh:
            sents = [ln.strip() for ln in fh if ln.strip()]

        changes: List[int] = []
        for a, b in zip(sents, sents[1:]):
            logits = model([a], [b])
            changes.append(int(logits.argmax(1).item()))

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": changes}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading model ...")
    model = load_model(Path(args.model_path), device,
                       n_proj=args.n_proj, seed=args.seed)

    predict_dir(model, Path(args.input_dir), Path(args.output_dir))

    print("solutions written to:", args.output_dir)


if __name__ == "__main__":
    main()
