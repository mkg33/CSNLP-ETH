#!/usr/bin/env python3
"""
evaluator for the max-sliced OTStyleModel

I used the following layout:

    pan25_data/
      ├─ easy/   ├─ validation/problem-*.txt
      │          └─ truth-problem-*.json
      ├─ medium/ etc
      └─ hard/   etc
"""

from __future__ import annotations
import argparse, json, os, random, sys
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
    """
    Max-sliced Wasserstein-1 distance.
    """
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
        self.n_proj = n_proj


        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder, add_prefix_space=True)
        self.encoder   = AutoModel.from_pretrained(encoder)

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
        dirs = dirs / dirs.norm(dim=1, keepdim=True)            # unit vectors
        self.register_buffer("proj_dirs", dirs)


    def _tokenize(self, s1: List[str], s2: List[str],
                  max_len: int = 128):
        enc = self.tokenizer(
            s1, s2,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
        token_type_ids = torch.zeros_like(enc["input_ids"])
        for i, spec_mask in enumerate(enc["special_tokens_mask"]):
            sep = (enc["input_ids"][i] == self.tokenizer.sep_token_id
                   ).nonzero(as_tuple=False)
            if sep.numel():
                token_type_ids[i, sep[0, 0] + 1:] = 1
        enc["token_type_ids"] = token_type_ids
        enc.pop("offset_mapping")
        enc.pop("special_tokens_mask")
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
            dists.append(max_sliced_w1_proj(
                H[b, a_idx], H[b, b_idx], self.proj_dirs))
        d_ot = torch.stack(dists).unsqueeze(1)

        logits = self.classifier(torch.cat([fused, d_ot], dim=1))
        return logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "max-sliced OT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="Model (.pt)")
    p.add_argument("--data-root", required=True, help="pan25_data dir")
    p.add_argument("--output-dir", required=True, help="output for solutions")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="use cpu/cuda")
    p.add_argument("--n-proj", type=int, default=128,
                   help="Number of random projections (must match training!!!)")
    p.add_argument("--seed", type=int, default=42,
                   help="Always 42.")
    return p.parse_args()


def load_model(path: str, device: torch.device,
               n_proj: int, seed: int) -> OTStyleModel:
    state = torch.load(path, map_location=device)

    c_dim = state["content_proj.weight"].shape[0]
    s_dim = state["style_proj.weight"].shape[0]

    model = OTStyleModel(content_dim=c_dim,
                         style_dim=s_dim,
                         n_proj=n_proj,
                         proj_seed=seed,
                         device=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_diff(model: OTStyleModel, diff: str,
                 data_root: str, out_root: str) -> None:
    inp_dir = Path(data_root) / diff / "validation"
    out_dir = Path(out_root) / diff
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in tqdm(sorted(inp_dir.glob("problem-*.txt")),
                         desc=diff, unit="file"):
        pid = txt_file.stem[8:]
        with txt_file.open(encoding="utf-8") as fh:
            sents = [ln.strip() for ln in fh if ln.strip()]

        changes: List[int] = []
        for s1, s2 in zip(sents, sents[1:]):
            logits = model([s1], [s2])
            changes.append(int(logits.argmax(dim=1).item()))

        (out_dir / f"solution-problem-{pid}.json").write_text(
            json.dumps({"changes": changes}, indent=2),
            encoding="utf-8")


def main() -> None:
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Model loading ...")
    model = load_model(args.model_path, device,
                       n_proj=args.n_proj, seed=args.seed)

    for diff in ("easy", "medium", "hard"):
        predict_diff(model, diff, args.data_root, args.output_dir)

    print("solutions written to:", args.output_dir)


if __name__ == "__main__":
    main()
